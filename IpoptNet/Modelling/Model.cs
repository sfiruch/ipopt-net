using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace IpoptNet.Modelling;

/// <summary>
/// A nonlinear-programming model: variables, constraints, objective, options, and (optionally)
/// implicit blocks. Call <see cref="Solve"/> to invoke IPOPT.
/// </summary>
/// <remarks>
/// Not thread-safe. A single Model instance must not have <see cref="Solve"/> called from
/// multiple threads concurrently — IPOPT itself is single-threaded per problem and the model's
/// per-pass scratch state is mutated during evaluation. Independent <see cref="Model"/> instances
/// can be solved on independent threads concurrently.
/// </remarks>
[DebuggerDisplay("Variables: {_variables.Count}, Constraints: {_constraints.Count}")]
public sealed partial class Model : IDisposable
{
    private readonly List<Variable> _variables = new();
    private readonly List<Constraint> _constraints = new();
    private readonly List<ImplicitBlock> _implicitBlocks = new();
    /// <summary>Union of all variables referenced (in raw mode) by any registered implicit
    /// block's residual. Maintained incrementally on each <see cref="AddImplicitBlock"/> so the
    /// topological-order check is O(new-block size) rather than O(blocks²).</summary>
    private readonly HashSet<Variable> _varsReferencedByBlocks = new();
    private Expr? _objective;
    private bool _disposed;

    /// <summary>
    /// When true, eliminated VariableNodes in this model behave as plain variables (read/write
    /// scratch[Variable.Index] directly) instead of redirecting through their <see cref="ImplicitBlock"/>.
    /// Set/cleared by <see cref="EnterRawMode"/> during a block's own Solve / sensitivity computation
    /// so residual expressions can be evaluated/differentiated without infinitely recursing.
    /// Per-Model rather than thread-static: each <see cref="Model"/> has its own flag, so
    /// independent models can be solved on independent threads concurrently without interference.
    /// </summary>
    internal bool IsRawMode { get; private set; }

    /// <summary>Generation counter for the per-pass expression value cache (see
    /// <see cref="ExprNode.Evaluate"/>). Bumped whenever the contents of the evaluation buffer
    /// change: at the start of each fresh IPOPT evaluation pass (SyncScratch) and around every
    /// implicit-block scratch mutation. A node's cached value is valid iff its stored generation
    /// equals this counter. Starts at 1 so nodes' initial generation 0 is always invalid.</summary>
    internal long EvalGeneration { get; private set; } = 1;


    /// <summary>True only while IPOPT evaluation callbacks may run (inside <see cref="Solve"/>).
    /// Outside of that window nodes always re-evaluate, so public Evaluate calls with arbitrary
    /// x vectors can never be served a stale cached value.</summary>
    internal bool ValueCachingActive { get; private set; }

    internal void InvalidateValueCache() => EvalGeneration++;

    /// <summary>Monotonic evaluation-pass id handed to <see cref="ImplicitBlock.Solve"/> so a block
    /// re-solves at most once per fresh pass. Deliberately instance state that is never reset:
    /// <c>ImplicitBlock._generation</c> is never reset either, so restarting this counter — between
    /// two <see cref="Solve"/> calls, or between two partitions of one partitioned solve — would
    /// make a block short-circuit on a stale generation and leave stale eliminated values in
    /// scratch.</summary>
    private long _evalPassCounter;

    /// <summary>IPOPT's convention for "no bound". Bounds are clamped to this magnitude.</summary>
    private const double Infinity = 1e19;

    /// <summary>IPOPT's own default for constr_viol_tol, used to decide whether an iterate counts as
    /// feasible when <see cref="IpoptOptions.ConstraintViolationTolerance"/> is left unset.</summary>
    private const double DefaultConstraintViolationTolerance = 1e-4;

    internal readonly struct RawModeScope : IDisposable
    {
        private readonly Model _model;
        private readonly bool _previous;
        public RawModeScope(Model model, bool previous) { _model = model; _previous = previous; }
        public void Dispose() => _model.IsRawMode = _previous;
    }

    internal RawModeScope EnterRawMode()
    {
        var prev = IsRawMode;
        IsRawMode = true;
        return new RawModeScope(this, prev);
    }

    /// <summary>
    /// IPOPT solver options. Configure before calling Solve().
    /// </summary>
    public IpoptOptions Options { get; } = new();

    /// <summary>
    /// Optional callback invoked at each IPOPT iteration.
    /// Return true to continue, false to request early termination.
    ///
    /// The <see cref="SolveStatistics"/> argument always describes the model as a whole. Under
    /// <see cref="EnablePartitioning"/> its <see cref="SolveStatistics.ObjectiveValue"/> is the
    /// full model objective — already-solved partitions at their optima, the partition currently
    /// iterating at its current iterate, not-yet-solved partitions at their start points — and its
    /// <see cref="SolveStatistics.IterationCount"/> is cumulative across partitions, so logic that
    /// tracks a best-so-far objective keeps working unchanged. The second argument says which
    /// sub-problem is iterating and carries the raw per-partition statistics in
    /// <see cref="PartitionInfo.LocalStatistics"/>; with partitioning off it reports
    /// <c>Index 0, Count 1</c>.
    ///
    /// Returning false stops the whole solve, exactly as it does without partitioning: the
    /// remaining partitions are not attempted and their variables are reported at their start
    /// values.
    /// </summary>
    public Func<SolveStatistics, PartitionInfo, bool>? IntermediateCallback { get; set; }

    public Model()
    {
    }

    public Variable AddVariable(double lowerBound = double.NegativeInfinity, double upperBound = double.PositiveInfinity)
    {
        var variable = new Variable(lowerBound, upperBound) { Index = _variables.Count };
        _variables.Add(variable);
        return variable;
    }

    public Variable AddVariable(double lowerBound, double upperBound, double scale = 1.0)
    {
        if (scale <= 0) throw new ArgumentException("Scale must be positive.", nameof(scale));
        var variable = new Variable(lowerBound, upperBound, scale) { Index = _variables.Count };
        _variables.Add(variable);
        return variable;
    }

    public Variable[] AddVariables(int x, double lowerBound, double upperBound, double scale = 1.0)
    {
        var res = new Variable[x];
        for (var i = 0; i < x; i++)
            res[i] = AddVariable(lowerBound, upperBound, scale);
        return res;
    }

    public Variable[,] AddVariables(int x, int y, double lowerBound, double upperBound)
    {
        var res = new Variable[x, y];
        for (var i = 0; i < x; i++)
            for (var j = 0; j < y; j++)
                res[i, j] = AddVariable(lowerBound, upperBound);
        return res;
    }

    public Variable[,] AddVariables(int x, int y, double lowerBound, double upperBound, double scale = 1.0)
    {
        var res = new Variable[x, y];
        for (var i = 0; i < x; i++)
            for (var j = 0; j < y; j++)
                res[i, j] = AddVariable(lowerBound, upperBound, scale);
        return res;
    }

    public Variable[,,] AddVariables(int x, int y, int z, double lowerBound, double upperBound, double scale = 1.0)
    {
        var res = new Variable[x, y, z];
        for (var i = 0; i < x; i++)
            for (var j = 0; j < y; j++)
                for (var k = 0; k < z; k++)
                    res[i, j, k] = AddVariable(lowerBound, upperBound, scale);
        return res;
    }

    public void SetObjective(Expr objective) => _objective = objective;

    public Constraint AddConstraint(Constraint constraint)
    {
        _constraints.Add(constraint);
        return constraint;
    }

    public Constraint AddConstraint(Expr expression, double lowerBound, double upperBound)
    {
        var c = new Constraint(expression, lowerBound, upperBound);
        _constraints.Add(c);
        return c;
    }

    /// <summary>
    /// Eliminates the listed variables from the IPOPT decision vector by treating the listed
    /// equality constraints as the implicit linear system A(other)·v = b(other) that defines
    /// them. The resulting NLP exposes only non-eliminated variables to IPOPT; eliminated values
    /// are recomputed numerically each evaluation pass via LU on a small dense matrix.
    /// First-order and second-order sensitivities propagate through the implicit-function theorem,
    /// so the caller can use either <see cref="HessianApproximation.Exact"/> or
    /// <see cref="HessianApproximation.LimitedMemory"/>.
    ///
    /// Constraints must be equalities (LowerBound == UpperBound = 0) and must be linear in the
    /// eliminated variables (they may be arbitrary in non-eliminated vars / parameters). The
    /// linearity check is verified numerically at the start of <see cref="Solve"/>.
    /// Eliminated variables must have infinite bounds — the block writes v* straight into the
    /// evaluation buffer, so a bound could only be violated silently. A non-unit
    /// <see cref="Variable.Scale"/> is fine.
    ///
    /// Blocks must be added in topological order: a block's residuals may only reference
    /// already-eliminated variables that belong to previously-added blocks. This is enforced
    /// at registration time.
    /// </summary>
    public void AddImplicitBlock(IReadOnlyList<Variable> variables, IReadOnlyList<Constraint> linearEqualities)
    {
        if (variables.Count == 0)
            throw new ArgumentException("At least one variable required.", nameof(variables));
        if (variables.Count != linearEqualities.Count)
            throw new ArgumentException(
                $"Number of variables ({variables.Count}) must equal number of equality constraints ({linearEqualities.Count}).");

        foreach (var v in variables)
        {
            if (v.Block is not null)
                throw new ArgumentException($"Variable x[{v.Index}] is already eliminated by another block.");
            if (!double.IsNegativeInfinity(v.LowerBound) || !double.IsPositiveInfinity(v.UpperBound))
                throw new ArgumentException(
                    $"Variable x[{v.Index}] has finite bounds (LB={v.LowerBound}, UB={v.UpperBound}). " +
                    "Bounds on eliminated variables are not supported.");
        }

        foreach (var c in linearEqualities)
        {
            if (c.LowerBound != c.UpperBound)
                throw new ArgumentException("All constraints in an implicit block must be equality (LowerBound == UpperBound).");
            if (c.LowerBound != 0)
                throw new ArgumentException("Implicit-block equality constraints must be of the form expression == 0 (LowerBound = UpperBound = 0).");
            if (!_constraints.Remove(c))
                throw new ArgumentException("Constraint not present in this model. Did you AddConstraint(...) it first?");
        }

        // Topological-order check: a variable about to be eliminated must not already appear in
        // any *earlier* implicit block's residual. If it did, that earlier block would solve
        // first (registration order) and capture v at a stale value while this block is what
        // actually defines v.
        // Regular constraints, the objective, and any other downstream expression are fine —
        // they evaluate v through VariableNode's redirect path, which gives the correct chain-rule
        // value. Only block residuals (which evaluate IN raw mode at scratch[v.Index]) are sensitive.
        // O(1)-per-variable thanks to the incremental cache (_varsReferencedByBlocks).
        foreach (var v in variables)
            if (_varsReferencedByBlocks.Contains(v))
                throw new ArgumentException(
                    $"Variable x[{v.Index}] cannot be eliminated: it is already referenced by an earlier implicit block's residual. " +
                    "Implicit blocks must be added in topological order — the block that defines a variable must be registered before any block whose residual reads it.");

        var varArr = variables.ToArray();
        var resArr = linearEqualities.Select(c => c.Expression).ToArray();
        var block = new ImplicitBlock(this, varArr, resArr);
        for (int j = 0; j < varArr.Length; j++)
        {
            varArr[j].Block = block;
            varArr[j].IndexInBlock = j;
        }
        _implicitBlocks.Add(block);

        // Extend the topological-order cache with the variables this new block's residuals
        // reference (raw mode so eliminated VariableNodes report themselves directly). Subsequent
        // AddImplicitBlock calls then check against this set in O(new-block size).
        using (EnterRawMode())
            foreach (var r in resArr)
                r.CollectVariables(_varsReferencedByBlocks);
    }

    public override string ToString()
    {
        var sb = new StringBuilder();

        sb.AppendLine($"Variables: {_variables.Count}");
        for (int i = 0; i < _variables.Count; i++)
        {
            var v = _variables[i];
            var bounds = "";
            if (v.LowerBound == v.UpperBound)
                bounds = $" == {v.LowerBound}";
            else if (v.LowerBound > double.NegativeInfinity && v.UpperBound < double.PositiveInfinity)
                bounds = $" in [{v.LowerBound}, {v.UpperBound}]";
            else if (v.LowerBound > double.NegativeInfinity)
                bounds = $" >= {v.LowerBound}";
            else if (v.UpperBound < double.PositiveInfinity)
                bounds = $" <= {v.UpperBound}";

            var start = v.Start.HasValue ? $", start={v.Start}" : "";
            var scale = v.Scale != 1.0 ? $", scale={v.Scale}" : "";
            var elim = v.IsEliminated ? " [eliminated]" : "";
            sb.AppendLine($"  x[{i}]{bounds}{start}{scale}{elim}");
        }

        sb.AppendLine();
        sb.AppendLine("Objective:");
        if (_objective is not null)
            sb.AppendLine($"  {_objective}");
        else
            sb.AppendLine("  (not set)");

        sb.AppendLine();
        sb.AppendLine($"Constraints: {_constraints.Count}");
        for (int i = 0; i < _constraints.Count; i++)
        {
            var c = _constraints[i];
            var boundsStr = "";
            if (c.LowerBound == c.UpperBound)
                boundsStr = $" == {c.LowerBound}";
            else if (c.LowerBound > double.NegativeInfinity && c.UpperBound < double.PositiveInfinity)
                boundsStr = $" in [{c.LowerBound}, {c.UpperBound}]";
            else if (c.LowerBound > double.NegativeInfinity)
                boundsStr = $" >= {c.LowerBound}";
            else if (c.UpperBound < double.PositiveInfinity)
                boundsStr = $" <= {c.UpperBound}";

            sb.AppendLine($"  Constraint[{i}]{boundsStr}: {c.Expression}");
        }

        if (_implicitBlocks.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine($"Implicit blocks: {_implicitBlocks.Count}");
            for (int b = 0; b < _implicitBlocks.Count; b++)
            {
                var block = _implicitBlocks[b];
                sb.AppendLine($"  Block[{b}]: {block.Variables.Length} eliminated var(s)");
            }
        }

        // Only when the caller opted in — the analysis is cheap but not free, and ToString() is a
        // debugger-display path that must never throw.
        if (EnablePartitioning && _objective is not null)
            try
            {
                var partitioning = AnalyzePartitions();
                sb.AppendLine();
                sb.AppendLine($"Partitions: {partitioning.Partitions.Count}");
                foreach (var p in partitioning.Partitions)
                    sb.AppendLine(ModelPartitioning.Describe(p));
            }
            catch (Exception ex)
            {
                sb.AppendLine();
                sb.AppendLine($"Partitions: (analysis failed: {ex.Message})");
            }

        return sb.ToString();
    }

    /// <summary>One sub-problem handed to a single IPOPT run. With partitioning off (or on a model
    /// that does not decompose) there is exactly one plan, covering the whole model.</summary>
    private sealed class SolvePlan
    {
        public required int Index;
        public required Variable[] ActiveVariables;      // non-eliminated, ascending Variable.Index
        public required Variable[] EliminatedVariables;  // ascending Variable.Index
        public required Constraint[] Constraints;        // model registration order
        public required ImplicitBlock[] Blocks;          // registration (topological) order
        public required Expr Objective;
        public required int[] CompactIndex;              // length totalVars; -1 outside this plan
    }

    /// <summary>Buffers sized by the whole model and shared by every plan. Allocated once per
    /// <see cref="Solve"/> call — re-allocating them per partition would eat part of the win
    /// partitioning exists to deliver.</summary>
    private sealed class SolveBuffers
    {
        public required double[] Scratch;          // indexed by Variable.Index, model-wide
        public required double[] BlockGradBuffer;  // for ImplicitBlock.Solve
        public required double[] FullGrad;         // objective gradient, original index space
        public required double[] JacGrad;          // constraint gradient, original index space
        public required int[] CompactToOriginal;
    }

    public ModelResult Solve(bool updateStartValues = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_objective is null)
            throw new InvalidOperationException("No objective function set");

        // Automatic elimination restructures the model — equalities move out of the constraint list
        // and into blocks — so everything downstream must see it in that shape. It is undone before
        // returning: EnableAutomaticElimination is an option for THIS solve, and a caller who turns
        // it off, or inspects the model afterwards, must find exactly what they built.
        var elimination = EnableAutomaticElimination ? ApplyAutomaticElimination() : null;
        try
        {
            return SolveCore(updateStartValues);
        }
        finally
        {
            elimination?.Undo();
        }
    }

    private ModelResult SolveCore(bool updateStartValues)
    {
        int totalVars = _variables.Count;
        var buffers = new SolveBuffers
        {
            // Per-evaluation scratch buffer (size = total vars, indexed by Variable.Index).
            // VariableNode.Evaluate reads from this; the model populates it from the IPOPT compact
            // vector before each evaluation pass and then runs every implicit block's Solve in order.
            // Every variable, eliminated or not, is held here in Scale-divided units.
            Scratch = new double[totalVars],
            BlockGradBuffer = new double[totalVars],
            FullGrad = new double[totalVars],
            JacGrad = new double[totalVars],
            CompactToOriginal = new int[totalVars],
        };

        // Prepare residual expressions inside each block (raw mode) before anything evaluates them:
        // the constant-constraint check below needs eliminated values, and so does every plan.
        foreach (var block in _implicitBlocks)
            block.PrepareResiduals();

        var constantConstraints = FindConstantConstraints();
        if (constantConstraints.Count > 0)
            ValidateConstantConstraints(constantConstraints, buffers);

        var plans = BuildSolvePlans(totalVars, out double objectiveConstant, out var unsolvedPlans, constantConstraints);

        // Each plan's objective and constraints in redirect mode — eliminated vars contribute their
        // block's transitive inputs. The model-level _objective is deliberately NOT prepared when it
        // was sliced: the slices cover every term between them, partitions are disjoint so no node is
        // reachable from two slices, and preparing the original as well would freeze its children
        // against a parent that is never evaluated.
        foreach (var plan in plans.Concat(unsolvedPlans))
        {
            plan.Objective.Prepare(this);
            foreach (var constraint in plan.Constraints)
                constraint.Expression.Prepare(this);
        }

        // Verify each block's residuals are linear in their own eliminated vars (fail fast on
        // misuse). Cheap: a couple of extra AccumulateGradient calls per residual at solve start.
        foreach (var block in _implicitBlocks)
            block.VerifyLinearity(totalVars);

        try
        {
            // The whole-model plan with nothing carved off: the pre-partitioning code path,
            // unchanged, which is what makes flag-on equivalent to flag-off on a model that does
            // not decompose.
            if (plans.Count == 1 && unsolvedPlans.Count == 0)
                return SolvePartition(plans[0], new PartitionContext(1, 0, 0, 0, 0, Normalize: false),
                    buffers, updateStartValues);

            // Seeds every variable's start point into scratch. Two uses: each partition's objective
            // at its start point (so the statistics handed to IntermediateCallback describe the whole
            // model from the very first iteration, rather than jumping each time a partition
            // completes), and the values reported for inert variables.
            SeedScratchFromStarts(buffers);

            // Partitions IPOPT never sees. Their variables take the seeded start point (for the
            // inert group) or whatever their blocks determined during the seeding pass (for the
            // all-eliminated ones), and their objective slice — a constant, since no decision
            // variable enters it — joins the model-level constant.
            var fixedSolution = new Dictionary<Variable, double>();
            foreach (var plan in unsolvedPlans)
            {
                foreach (var v in plan.ActiveVariables)
                    fixedSolution[v] = buffers.Scratch[v.Index] * v.Scale;
                foreach (var v in plan.EliminatedVariables)
                    fixedSolution[v] = buffers.Scratch[v.Index] * v.Scale;
                objectiveConstant += plan.Objective.Evaluate(buffers.Scratch);
            }

            var startObjective = plans.Select(p => p.Objective.Evaluate(buffers.Scratch)).ToArray();

            var results = new List<ModelResult>(plans.Count);
            double completedObjective = 0;
            int completedIterations = 0;
            SolveStatistics? completedStatistics = null;

            // Time limits are model-wide deadlines, so each partition gets what is left rather than
            // the full budget: N partitions must not be able to take N times as long as the caller
            // allowed. Iteration limits are deliberately NOT shared — max_iter is a "don't spin
            // forever on this sub-problem" guard, and splitting it across partitions would make a
            // later partition fail for having been preceded by a hard one.
            var wallClock = Options.MaxWallTime is not null ? Stopwatch.StartNew() : null;
            var cpuAtStart = Options.MaxCpuTime is not null ? Process.GetCurrentProcess().TotalProcessorTime : default;

            for (int p = 0; p < plans.Count; p++)
            {
                double pending = 0;
                for (int q = p + 1; q < plans.Count; q++)
                    pending += startObjective[q];

                double? remainingWall = Options.MaxWallTime is { } maxWall
                    ? Math.Max(maxWall - wallClock!.Elapsed.TotalSeconds, ExhaustedTimeBudgetSeconds)
                    : null;
                double? remainingCpu = Options.MaxCpuTime is { } maxCpu
                    ? Math.Max(maxCpu - (Process.GetCurrentProcess().TotalProcessorTime - cpuAtStart).TotalSeconds,
                               ExhaustedTimeBudgetSeconds)
                    : null;

                // Every partition is attempted: a failure in one says nothing about the others, and
                // callers rely on their per-iteration callback seeing all of them.
                var result = SolvePartition(plans[p],
                    new PartitionContext(plans.Count, completedObjective, pending, objectiveConstant,
                        completedIterations, Normalize: true, remainingWall, remainingCpu, completedStatistics),
                    buffers, updateStartValues);
                results.Add(result);
                completedObjective += result.ObjectiveValue;
                completedIterations += result.Statistics.IterationCount;
                completedStatistics = MergeStatistics(completedStatistics, result.Statistics);

                // An explicit stop request means stop, exactly as it would without partitioning.
                if (result.Status == ApplicationReturnStatus.UserRequestedStop)
                    break;
            }

            return CombineResults(results, plans, buffers, objectiveConstant, fixedSolution);
        }
        finally
        {
            // Clear cached variables to free memory after optimization. _objective too: harmless
            // when it was never prepared, and necessary when it is itself a plan's objective.
            foreach (var plan in plans.Concat(unsolvedPlans))
            {
                plan.Objective.Clear();
                foreach (var constraint in plan.Constraints)
                    constraint.Expression.Clear();
            }
            _objective!.Clear();   // Solve() guaranteed non-null before delegating here
            foreach (var block in _implicitBlocks)
                block.ClearResiduals();
        }
    }

    /// <summary>Constraints whose expression references no variable at all, so their value is fixed
    /// before the solve begins. The usual source is a bound on a variable an implicit block pins to a
    /// constant: in redirect mode an eliminated variable reports its block's inputs, and a block with
    /// no inputs reports none. IPOPT cannot be given such a constraint — the row's Jacobian is empty,
    /// which the C API rejects outright when it is the only row and trips a missing-key lookup in the
    /// Jacobian callback otherwise — so they are checked once and then left out of the problem.</summary>
    private List<Constraint> FindConstantConstraints()
    {
        var found = new List<Constraint>();
        var vars = new HashSet<Variable>();
        foreach (var c in _constraints)
        {
            vars.Clear();
            c.Expression.CollectVariables(vars);
            if (vars.Count == 0)
                found.Add(c);
        }
        return found;
    }

    /// <summary>Evaluates the constant constraints once and rejects the model if any cannot hold.
    /// No choice of decision variables can affect them, so this is decided before the search starts
    /// rather than discovered by it — a thrown error naming the offender is more use than an
    /// infeasible status the caller has to go and diagnose.</summary>
    private void ValidateConstantConstraints(List<Constraint> constantConstraints, SolveBuffers buffers)
    {
        SeedScratchFromStarts(buffers);
        double tolerance = Options.ConstraintViolationTolerance ?? DefaultConstraintViolationTolerance;

        foreach (var c in constantConstraints)
        {
            double value = c.Expression.Evaluate(buffers.Scratch);
            if (value >= c.LowerBound - tolerance && value <= c.UpperBound + tolerance)
                continue;

            string bounds = c.LowerBound == c.UpperBound ? $"== {c.LowerBound}"
                : double.IsNegativeInfinity(c.LowerBound) ? $"<= {c.UpperBound}"
                : double.IsPositiveInfinity(c.UpperBound) ? $">= {c.LowerBound}"
                : $"in [{c.LowerBound}, {c.UpperBound}]";
            throw new InvalidOperationException(
                $"Constraint {_constraints.IndexOf(c)} ({c.Expression}) references no decision variable, so its "
                + $"value is fixed at {value}, which does not satisfy {bounds}. No solve can change this — the "
                + "constraint depends only on constants and on variables pinned by implicit blocks.");
        }
    }

    /// <summary>Fills scratch with every variable's start point — the same defaulting rule the
    /// per-partition solve applies to its IPOPT <c>x</c> vector — and runs every block once so
    /// eliminated values are populated too.</summary>
    private void SeedScratchFromStarts(SolveBuffers buffers)
    {
        foreach (var v in _variables)
        {
            if (v.IsEliminated) continue;
            double lo = Math.Clamp(v.LowerBound / v.Scale, -Infinity, Infinity);
            double hi = Math.Clamp(v.UpperBound / v.Scale, -Infinity, Infinity);
            buffers.Scratch[v.Index] =
                v.Start.HasValue ? Math.Clamp(v.Start.Value / v.Scale, lo, hi) :
                hi == Infinity ? Math.Max(0, lo) :
                lo == -Infinity ? Math.Min(0, hi) :
                (lo + hi) * 0.5;
        }

        _evalPassCounter++;
        InvalidateValueCache();
        foreach (var block in _implicitBlocks)
            block.Solve(buffers.Scratch, _evalPassCounter, buffers.BlockGradBuffer);
    }

    /// <summary>Builds the per-partition solve plans. Returns a single whole-model plan (and no
    /// inert variables) when <see cref="EnablePartitioning"/> is off or the model does not
    /// decompose — that path is the pre-partitioning code, unchanged, which is what makes flag-on
    /// equivalent to flag-off.</summary>
    /// <param name="totalVars">Size of the model-wide variable index space.</param>
    /// <param name="objectiveConstant">The objective's additive constant, held out of every slice and
    /// added back once during aggregation. Zero on the whole-model path, where the objective keeps it.</param>
    /// <param name="unsolvedPlans">Partitions that are not handed to IPOPT, because there is nothing
    /// for it to decide: the inert group (referenced by nothing at all), and any partition whose
    /// variables are all eliminated by implicit blocks, which leaves a zero-variable NLP that IPOPT
    /// refuses to create. The caller resolves their variables from the seeded start point and folds
    /// their objective slice — a constant, by construction — into the model total.</param>
    /// <param name="constantConstraints">Constraints that reference no decision variable, already
    /// validated by the caller. They are left out of every plan: IPOPT cannot act on them.</param>
    private List<SolvePlan> BuildSolvePlans(int totalVars, out double objectiveConstant,
        out List<SolvePlan> unsolvedPlans, List<Constraint> constantConstraints)
    {
        objectiveConstant = 0;
        unsolvedPlans = [];

        if (EnablePartitioning)
        {
            var layout = ComputePartitionLayout();
            if (layout.Count > 1)
            {
                objectiveConstant = ObjectiveConstantTerm;
                var plans = new List<SolvePlan>(layout.Count);
                for (int p = 0; p < layout.Count; p++)
                {
                    // Nothing for IPOPT to decide. Either the inert group — free variables the model
                    // references nowhere — or a partition whose variables are all determined by its
                    // implicit blocks, which would be a zero-variable NLP that IPOPT refuses to
                    // create. Constraints, though, must still be judged by someone: refuse rather
                    // than silently drop them.
                    if (layout.IsInert[p] || layout.ActiveVariables[p].Length == 0)
                    {
                        // Such a partition never carries constraints, so nothing is dropped by not
                        // solving it. A constraint's collected variables are always free ones
                        // (AnalyzeJacobianSparsity rejects eliminated ones), so a constraint with any
                        // variables puts a free variable in its partition; and a constraint with none
                        // set decomposable = false, collapsing the model to a single partition that
                        // never reaches this branch.
                        Debug.Assert(layout.Constraints[p].Length == 0,
                            "A partition with no free variables cannot carry constraints.");

                        unsolvedPlans.Add(new SolvePlan
                        {
                            Index = unsolvedPlans.Count,
                            ActiveVariables = layout.ActiveVariables[p],
                            EliminatedVariables = layout.EliminatedVariables[p],
                            Constraints = [],
                            Blocks = layout.Blocks[p],
                            Objective = BuildPartitionObjective(layout.ObjectiveTerms[p]),
                            CompactIndex = [],
                        });
                        continue;
                    }

                    var compactIndex = new int[totalVars];
                    Array.Fill(compactIndex, -1);
                    for (int i = 0; i < layout.ActiveVariables[p].Length; i++)
                        compactIndex[layout.ActiveVariables[p][i].Index] = i;

                    plans.Add(new SolvePlan
                    {
                        Index = plans.Count,
                        ActiveVariables = layout.ActiveVariables[p],
                        EliminatedVariables = layout.EliminatedVariables[p],
                        Constraints = layout.Constraints[p],
                        Blocks = layout.Blocks[p],
                        Objective = BuildPartitionObjective(layout.ObjectiveTerms[p]),
                        CompactIndex = compactIndex,
                    });
                }
                return plans;
            }
        }

        // Whole-model plan. Eliminated variables get compact index -1 and are not exposed to IPOPT.
        var wholeCompactIndex = new int[totalVars];
        var activeVars = new List<Variable>(totalVars);
        var eliminatedVars = new List<Variable>();
        for (int i = 0; i < totalVars; i++)
        {
            if (_variables[i].IsEliminated)
            {
                wholeCompactIndex[i] = -1;
                eliminatedVars.Add(_variables[i]);
            }
            else
            {
                wholeCompactIndex[i] = activeVars.Count;
                activeVars.Add(_variables[i]);
            }
        }

        return
        [
            new SolvePlan
            {
                Index = 0,
                ActiveVariables = [.. activeVars],
                EliminatedVariables = [.. eliminatedVars],
                Constraints = [.. _constraints.Where(c => !constantConstraints.Contains(c))],
                Blocks = [.. _implicitBlocks],
                Objective = _objective!,
                CompactIndex = wholeCompactIndex,
            }
        ];
    }

    /// <summary>Where one sub-problem sits within the overall solve, so its IPOPT statistics can be
    /// reported as model-level quantities. <paramref name="Normalize"/> is false only on the
    /// untouched whole-model path, where the raw statistics already describe the whole model.</summary>
    private readonly record struct PartitionContext(
        int Count,
        double CompletedObjective,
        double PendingObjective,
        double ObjectiveConstant,
        int CompletedIterations,
        bool Normalize,
        double? RemainingWallTime = null,
        double? RemainingCpuTime = null,
        SolveStatistics? Completed = null);

    /// <summary>Floor for a per-partition time budget. IPOPT rejects a non-positive limit, so an
    /// exhausted budget is handed over as this instead — small enough that the partition stops
    /// almost immediately with MaximumWallTimeExceeded / MaximumCpuTimeExceeded.</summary>
    private const double ExhaustedTimeBudgetSeconds = 1e-3;

    private ModelResult SolvePartition(SolvePlan plan, PartitionContext ctx, SolveBuffers buffers,
        bool updateStartValues)
    {
        // Exact Hessian propagation through implicit blocks is implemented via VariableNode
        // (PropagateHessian) plus the QuadExprNode cross-product term. Caller chooses Hessian mode.

        int totalVars = buffers.Scratch.Length;
        var activeVars = plan.ActiveVariables;
        var compactIndex = plan.CompactIndex;
        int n = activeVars.Length;
        int m = plan.Constraints.Length;

        // Variable bounds (active vars only; divided by Scale so IPOPT works with normalized internal variables)
        var xL = new double[n];
        var xU = new double[n];
        for (int i = 0; i < n; i++)
        {
            xL[i] = Math.Clamp(activeVars[i].LowerBound / activeVars[i].Scale, -Infinity, Infinity);
            xU[i] = Math.Clamp(activeVars[i].UpperBound / activeVars[i].Scale, -Infinity, Infinity);
        }

        // Constraint bounds
        var gL = new double[m];
        var gU = new double[m];
        for (int i = 0; i < m; i++)
        {
            gL[i] = plan.Constraints[i].LowerBound;
            gU[i] = plan.Constraints[i].UpperBound;
        }

        var scratch = buffers.Scratch;

        // Helper: synchronize scratch with the IPOPT compact x and run this plan's blocks. Other
        // plans' variables and blocks are left untouched — nothing in this sub-problem reads them.
        void SyncScratch(ReadOnlySpan<double> compactX)
        {
            _evalPassCounter++;
            for (int i = 0; i < n; i++)
                scratch[activeVars[i].Index] = compactX[i];
            InvalidateValueCache();
            foreach (var block in plan.Blocks)
                block.Solve(scratch, _evalPassCounter, buffers.BlockGradBuffer);
        }

        // Analyze sparsity (in compact column space)
        var (jacRows, jacCols) = AnalyzeJacobianSparsity(plan.Constraints, compactIndex);

        var useLimitedMemory = Options.HessianApproximation == HessianApproximation.LimitedMemory;
        var (hessRowsOrig, hessColsOrig) = useLimitedMemory ? (Array.Empty<int>(), Array.Empty<int>()) : AnalyzeHessianSparsity(plan.Objective, plan.Constraints);
        // IPOPT-facing iRow/jCol must be in compact column space (active-variable indexing).
        var hessRows = new int[hessRowsOrig.Length];
        var hessCols = new int[hessColsOrig.Length];
        for (int i = 0; i < hessRowsOrig.Length; i++)
        {
            hessRows[i] = compactIndex[hessRowsOrig[i]];
            hessCols[i] = compactIndex[hessColsOrig[i]];
            // An eliminated variable, or — under partitioning — a variable outside this partition.
            // The latter cannot happen for a correct decomposition: every Hessian entry comes from a
            // constraint or objective term whose whole variable set was unioned into one partition.
            // So this stays a loud invariant check rather than a silent filter, which would drop
            // Hessian entries and yield wrong-but-plausible answers.
            if (hessRows[i] < 0 || hessCols[i] < 0)
                throw new InvalidOperationException(
                    $"Hessian sparsity entry ({hessRowsOrig[i]}, {hessColsOrig[i]}) in partition {plan.Index} references " +
                    "an eliminated variable or one outside the partition. CollectHessianSparsity must only return " +
                    "non-eliminated variable indices, and partition analysis must keep coupled variables together.");
        }

        // With eliminated states, what IPOPT asks for are *reduced* derivatives, and computing them
        // needs the objective and constraints differentiated with respect to the states as well as
        // the parameters — that is, walked in raw mode. The IPOPT-facing sparsity above had to come
        // from the redirect-mode walk, since it is stated in parameter space, so the switch to raw
        // mode happens here: once, after the structure is fixed and before anything evaluates.
        ReducedDerivatives? reduced = null;
        if (plan.Blocks.Length > 0)
        {
            plan.Objective.Clear();
            foreach (var constraint in plan.Constraints)
                constraint.Expression.Clear();
            using (EnterRawMode())
            {
                plan.Objective.Prepare(this);
                foreach (var constraint in plan.Constraints)
                    constraint.Expression.Prepare(this);
                reduced = new ReducedDerivatives(this, plan.Blocks, activeVars, plan.Objective,
                    plan.Constraints, scratch);
            }
        }

        // Create callbacks
        var evalF = CreateEvalFCallback(plan.Objective, scratch, SyncScratch);
        var evalGradF = CreateEvalGradFCallback(plan.Objective, scratch, SyncScratch, totalVars, compactIndex, buffers.FullGrad, reduced);
        var evalG = CreateEvalGCallback(plan.Constraints, scratch, SyncScratch);
        var evalJacG = CreateEvalJacGCallback(jacRows, jacCols, plan.Constraints, scratch, SyncScratch, totalVars, compactIndex, buffers.CompactToOriginal, buffers.JacGrad, reduced);
        var evalH = useLimitedMemory ? CreateDummyEvalHCallback() : CreateEvalHCallback(hessRowsOrig, hessColsOrig, hessRows, hessCols, plan.Objective, plan.Constraints, scratch, SyncScratch, totalVars, reduced);

        ApplicationReturnStatus status;
        double objValue;
        SolveStatistics statistics;
        var constraintValues = new double[m];
        var constraintMultipliers = new double[m];
        var lowerBoundMultipliers = new double[n];
        var upperBoundMultipliers = new double[n];

        using var solver = new IpoptSolver(
            n, xL, xU,
            m, gL, gU,
            jacRows.Length, hessRows.Length,
            evalF, evalGradF, evalG, evalJacG, evalH);

        // Best-so-far tracking. IPOPT returns its FINAL iterate, which is not always its best one: a
        // run ending on MaximumIterationsExceeded, RestorationFailed or a caller stop can leave a
        // worse point than it passed through earlier. Feasibility comes first — a low objective at a
        // point that violates the constraints is not a better answer.
        var bestX = new double[n];
        var candidateX = new double[n];
        double bestObjective = 0, bestViolation = 0;
        bool bestIsFeasible = false, haveBest = false;
        int bestIteration = -1;
        double violationTolerance = Options.ConstraintViolationTolerance ?? DefaultConstraintViolationTolerance;

        // The statistics the caller sees always describe the whole model, so best-so-far tracking in
        // caller code keeps working under partitioning; PartitionInfo carries the raw per-partition
        // values. Installed unconditionally — the iterate tracking above needs it even when the
        // caller has no callback of their own.
        solver.IntermediateCallback = local =>
        {
            // Restoration-phase iterates describe IPOPT's internal feasibility-restoration NLP, not
            // ours: its objective is a different quantity on a different scale, so comparing it to
            // our incumbent would be meaningless. Skip them; the regular-mode iterates either side
            // of a restoration episode are still considered.
            if (local.AlgorithmMode == AlgorithmMode.RegularMode && solver.TryGetCurrentIterate(candidateX))
            {
                bool feasible = local.PrimalInfeasibility <= violationTolerance;
                bool better =
                    !haveBest ? true :
                    feasible && !bestIsFeasible ? true :                                  // any feasible beats any infeasible
                    feasible && bestIsFeasible ? local.ObjectiveValue < bestObjective :    // both feasible: lower objective
                    !bestIsFeasible ? local.PrimalInfeasibility < bestViolation :          // neither: closer to feasible
                    false;                                                                 // infeasible cannot beat feasible
                if (better)
                {
                    candidateX.CopyTo(bestX, 0);
                    bestObjective = local.ObjectiveValue;
                    bestViolation = local.PrimalInfeasibility;
                    bestIsFeasible = feasible;
                    bestIteration = local.IterationCount;
                    haveBest = true;
                }
            }

            if (IntermediateCallback is not { } userCallback)
                return true;

            // Everything the caller sees describes the whole model. Folding in the partitions already
            // finished matters for more than tidiness: a consumer that latches the last callback's
            // PrimalInfeasibility to judge the solve would otherwise read only the LAST partition's
            // value and miss an infeasible one solved earlier.
            var reported = !ctx.Normalize ? local : MergeStatistics(ctx.Completed, local) with
            {
                ObjectiveValue = ctx.CompletedObjective + local.ObjectiveValue + ctx.PendingObjective + ctx.ObjectiveConstant,
            };
            return userCallback(reported, new PartitionInfo(plan.Index, ctx.Count, plan.ActiveVariables, m, local));
        };

        // Apply user-specified options
        foreach (var (name, value) in Options.Options)
        {
            bool ok = value switch
            {
                string strValue => solver.SetOption(name, strValue),
                int intValue    => solver.SetOption(name, intValue),
                double dblValue => solver.SetOption(name, dblValue),
                _               => true
            };
            if (!ok)
                throw new InvalidOperationException($"IPOPT rejected option '{name}' = '{value}'. Check that the option name and value are valid.");
        }

        // Model-wide time budgets: override whatever the user-option loop just applied with what is
        // left of the deadline. Only on the partitioned path — the whole-model path must stay
        // byte-identical to the pre-partitioning code.
        if (ctx.RemainingWallTime is { } remainingWall)
            if (!solver.SetOption("max_wall_time", remainingWall))
                throw new InvalidOperationException($"IPOPT rejected option 'max_wall_time' = '{remainingWall}'.");
        if (ctx.RemainingCpuTime is { } remainingCpu)
            if (!solver.SetOption("max_cpu_time", remainingCpu))
                throw new InvalidOperationException($"IPOPT rejected option 'max_cpu_time' = '{remainingCpu}'.");

        // Each partition runs its own IpoptSolver, so without file_append every partition after the
        // first truncates output_file and only the last one's log survives. An explicit
        // Options.FileAppend is honoured — the caller opted into whatever it says.
        if (plan.Index > 0 && Options.FileAppend is null && Options.OutputFile is not null)
            if (!solver.SetOption("file_append", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'file_append' = 'yes'.");

        // Auto-enable warm start if we have non-zero dual values and user hasn't explicitly set it.
        // Never on a constraint-free sub-problem: IPOPT answers warm_start_init_point=yes with
        // UnrecoverableException when m == 0 (reproducible on a single solve, no implicit blocks
        // involved). That is easy to walk into — any second Solve() has written back non-zero bound
        // duals, and partitioning routinely produces partitions whose variables carry only bounds.
        // An explicit Options.WarmStartInitPoint is still honoured; only this heuristic steps aside.
        if (Options.WarmStartInitPoint is null && m > 0 &&
            (activeVars.Any(v => v.LowerBoundDualStart != 0 || v.UpperBoundDualStart != 0) ||
             plan.Constraints.Any(c => c.DualStart != 0)))
        {
            if (!solver.SetOption("warm_start_init_point", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'warm_start_init_point' = 'yes'.");
        }

        // Auto-enable grad_f_constant if objective has constant gradients and user hasn't explicitly set it
        if (Options.GradFConstant is null && plan.Objective.IsLinear())
            if (!solver.SetOption("grad_f_constant", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'grad_f_constant' = 'yes'.");

        // Auto-enable jac_c_constant if all equality constraints have constant Jacobians.
        // Note: when implicit blocks are present, "linear" via VariableNode for an eliminated var
        // returns false (since v* depends nonlinearly on inputs). So this is automatically skipped.
        var equalityConstraints = plan.Constraints.Where(c => Math.Abs(c.LowerBound - c.UpperBound) < 1e-15).ToList();
        if (Options.JacCConstant is null && equalityConstraints.All(c => c.Expression.IsLinear()))
            if (!solver.SetOption("jac_c_constant", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'jac_c_constant' = 'yes'.");

        var inequalityConstraints = plan.Constraints.Where(c => Math.Abs(c.LowerBound - c.UpperBound) >= 1e-15).ToList();
        if (Options.JacDConstant is null && inequalityConstraints.All(c => c.Expression.IsLinear()))
            if (!solver.SetOption("jac_d_constant", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'jac_d_constant' = 'yes'.");

        if (Options.HessianConstant is null && !useLimitedMemory &&
            plan.Objective.IsAtMostQuadratic() && plan.Constraints.All(c => c.Expression.IsLinear()))
        {
            if (!solver.SetOption("hessian_constant", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'hessian_constant' = 'yes'.");
        }

        // Initialize primal variables from variable Start values, ensuring they're within bounds
        var x = new double[n];
        for (int i = 0; i < n; i++)
            if (activeVars[i].Start.HasValue)
                x[i] = Math.Clamp(activeVars[i].Start!.Value / activeVars[i].Scale, xL[i], xU[i]);
            else if (xU[i] == Infinity)
                x[i] = Math.Max(0, xL[i]);
            else if (xL[i] == -Infinity)
                x[i] = Math.Min(0, xU[i]);
            else
            {
                Debug.Assert(xU[i] != Infinity && xL[i] != -Infinity);
                x[i] = (xL[i] + xU[i]) * 0.5;
            }

        // Initialize dual variables
        for (int i = 0; i < m; i++)
            constraintMultipliers[i] = plan.Constraints[i].DualStart;

        for (int i = 0; i < n; i++)
        {
            lowerBoundMultipliers[i] = activeVars[i].LowerBoundDualStart;
            upperBoundMultipliers[i] = activeVars[i].UpperBoundDualStart;
        }

        // Enable the per-pass expression value cache only while IPOPT callbacks can run. Everything
        // before (VerifyLinearity's own scratch) and after (final SyncScratch, solution readback)
        // evaluates uncached, so no cache-invalidation discipline is needed there.
        ValueCachingActive = true;
        try
        {
            status = solver.Solve(x, out objValue, out statistics, constraintValues, constraintMultipliers,
                                      lowerBoundMultipliers, upperBoundMultipliers);
        }
        finally
        {
            ValueCachingActive = false;
        }

        // Reconstruct the best iterate's full variable values. Done before the final readback so
        // that readback's own SyncScratch leaves scratch describing the returned solution, which is
        // what CombineResults expects. Eliminated variables are recomputed by running this
        // partition's blocks at the best x — the same path the final readback uses, and guarded the
        // same way, since a pathological iterate can make a block's LU singular.
        IterateSnapshot? bestIterate = null;
        if (haveBest)
            try
            {
                SyncScratch(bestX.AsSpan());
                var bestSolution = new Dictionary<Variable, double>();
                for (int i = 0; i < n; i++)
                    bestSolution[activeVars[i]] = Math.Clamp(bestX[i], xL[i], xU[i]) * activeVars[i].Scale;
                foreach (var v in plan.EliminatedVariables)
                    bestSolution[v] = scratch[v.Index] * v.Scale;
                bestIterate = new IterateSnapshot(bestSolution, bestObjective, bestViolation, bestIsFeasible, bestIteration);
            }
            catch (InvalidOperationException)
            {
                bestIterate = null;
            }

        // Build solution. We expose all variables (including eliminated ones) in the dictionary
        // so callers see consistent values; eliminated vars are read from scratch after a final
        // sync at the returned x. Wrapped in try/catch: on pathological exit statuses (e.g.
        // InvalidNumberDetected, RestorationFailed) the returned x can contain NaN, which would
        // make SyncScratch's per-block LU singular and throw. Prefer null Solution over crashing
        // the whole Solve call in that case.
        Dictionary<Variable, double>? solution = null;
        try
        {
            solution = new Dictionary<Variable, double>();
            // Sync once more to populate scratch with eliminated values at the final iterate.
            SyncScratch(x.AsSpan());
            for (int i = 0; i < n; i++)
                solution[activeVars[i]] = Math.Clamp(x[i], xL[i], xU[i]) * activeVars[i].Scale;
            foreach (var v in plan.EliminatedVariables)
                solution[v] = scratch[v.Index] * v.Scale;
        }
        catch (InvalidOperationException)
        {
            // Singular implicit block at the final iterate; iterate is unusable.
            solution = null;
        }

        // Update variable Start values and dual variables if requested and solution is usable.
        if (updateStartValues && solution is not null && status is
            ApplicationReturnStatus.SolveSucceeded or
            ApplicationReturnStatus.SolvedToAcceptableLevel or
            ApplicationReturnStatus.FeasiblePointFound or
            ApplicationReturnStatus.InfeasibleProblemDetected or
            ApplicationReturnStatus.SearchDirectionBecomesTooSmall or
            ApplicationReturnStatus.UserRequestedStop or
            ApplicationReturnStatus.MaximumIterationsExceeded or
            ApplicationReturnStatus.MaximumCpuTimeExceeded or
            ApplicationReturnStatus.MaximumWallTimeExceeded or
            ApplicationReturnStatus.RestorationFailed)
        {
            for (int i = 0; i < n; i++)
            {
                activeVars[i].Start = solution[activeVars[i]];
                activeVars[i].LowerBoundDualStart = lowerBoundMultipliers[i];
                activeVars[i].UpperBoundDualStart = upperBoundMultipliers[i];
            }
            foreach (var v in plan.EliminatedVariables)
                v.Start = solution[v];

            for (int i = 0; i < m; i++)
                plan.Constraints[i].DualStart = constraintMultipliers[i];
        }

        return new ModelResult(status, solution, objValue, statistics) { BestIterate = bestIterate };
    }

    private static (int[] rows, int[] cols) AnalyzeJacobianSparsity(Constraint[] constraints, int[] compactIndex)
    {
        var entries = new HashSet<(int row, int col)>();
        var vars = new HashSet<Variable>();

        for (int i = 0; i < constraints.Length; i++)
        {
            vars.Clear();
            constraints[i].Expression.CollectVariables(vars);
            foreach (var v in vars)
            {
                if (v.IsEliminated)
                    throw new InvalidOperationException(
                        $"Constraint {i} CollectVariables returned eliminated variable x[{v.Index}]. " +
                        "Did the implicit block's CollectInputVariables not get called?");
                entries.Add((i, compactIndex[v.Index]));
            }
        }

        var entriesArray = new (int row, int col)[entries.Count];
        entries.CopyTo(entriesArray);
        Array.Sort(entriesArray, (a, b) =>
        {
            int cmp = a.row.CompareTo(b.row);
            return cmp != 0 ? cmp : a.col.CompareTo(b.col);
        });

        var rows = new int[entriesArray.Length];
        var cols = new int[entriesArray.Length];
        for (int i = 0; i < entriesArray.Length; i++)
        {
            rows[i] = entriesArray[i].row;
            cols[i] = entriesArray[i].col;
        }
        return (rows, cols);
    }

    /// <summary>Returns Hessian sparsity in ORIGINAL Variable.Index space (for use with the
    /// internal HessianAccumulator which gets hess.Add(origIdx, origIdx, value) calls). The
    /// caller is responsible for remapping to compact when reporting iRow/jCol to IPOPT.</summary>
    private static (int[] rows, int[] cols) AnalyzeHessianSparsity(Expr objective, Constraint[] constraints)
    {
        var entries = new HashSet<(int row, int col)>();
        objective.CollectHessianSparsity(entries);
        foreach (var c in constraints)
            c.Expression.CollectHessianSparsity(entries);

        var sortedEntries = entries.OrderBy(e => e.row).ThenBy(e => e.col).ToList();

        var rows = new int[sortedEntries.Count];
        var cols = new int[sortedEntries.Count];
        for (int i = 0; i < sortedEntries.Count; i++)
        {
            rows[i] = sortedEntries[i].row;
            cols[i] = sortedEntries[i].col;
        }
        return (rows, cols);
    }

    private static unsafe EvalFCallback CreateEvalFCallback(Expr objective, double[] scratch, Action<ReadOnlySpan<double>> syncScratch)
    {
        return (int n, double* pX, bool newX, double* objValue, nint userData) =>
        {
            var x = new ReadOnlySpan<double>(pX, n);
            if (newX) syncScratch(x);
            *objValue = objective.Evaluate(scratch);
            return IsValidNumber(*objValue);
        };
    }

    private static unsafe EvalGradFCallback CreateEvalGradFCallback(Expr objective, double[] scratch, Action<ReadOnlySpan<double>> syncScratch, int totalVars, int[] compactIndex, double[] fullGrad, ReducedDerivatives? reduced)
    {
        return (int n, double* pX, bool newX, double* pGradF, nint userData) =>
        {
            var x = new ReadOnlySpan<double>(pX, n);
            if (newX) syncScratch(x);
            var gradF = new Span<double>(pGradF, n);
            gradF.Clear();
            if (reduced is not null)
            {
                // F_p + Xᵀ F_v, written straight into compact column space.
                reduced.Gradient(objective, gradF);
            }
            else
            {
            // The objective's _cachedVariables only includes non-eliminated vars (CollectVariables
            // in redirect mode walks blocks). AccumulateGradient writes into fullGrad indexed by
            // Variable.Index. We re-pack into the IPOPT compact gradF using compactIndex.
            Array.Clear(fullGrad);
            objective.AccumulateGradient(scratch, fullGrad);
            for (int i = 0; i < totalVars; i++)
            {
                int ci = compactIndex[i];
                if (ci < 0) continue;
                gradF[ci] = fullGrad[i];
            }
            }

            for (int i = 0; i < n; i++)
                if (!IsValidNumber(gradF[i]))
                    return false;

            return true;
        };
    }

    public static bool IsValidNumber(double v) => !double.IsInfinity(v) && !double.IsNaN(v);

    private static unsafe EvalGCallback CreateEvalGCallback(Constraint[] constraints, double[] scratch, Action<ReadOnlySpan<double>> syncScratch)
    {
        return (int n, double* pX, bool newX, int m, double* pG, nint userData) =>
        {
            var x = new ReadOnlySpan<double>(pX, n);
            if (newX) syncScratch(x);
            var g = new Span<double>(pG, m);
            for (int i = 0; i < m; i++)
            {
                g[i] = constraints[i].Expression.Evaluate(scratch);
                if (!IsValidNumber(g[i]))
                    return false;
            }
            return true;
        };
    }

    private static unsafe EvalJacGCallback CreateEvalJacGCallback(int[] structRows, int[] structCols, Constraint[] constraints, double[] scratch, Action<ReadOnlySpan<double>> syncScratch, int totalVars, int[] compactIndex, int[] compactToOriginal, double[] grad, ReducedDerivatives? reduced)
    {
        // structCols are in COMPACT column space. To map a constraint's gradient (computed in
        // original-variable-index space) into the right value slots, we precompute an inverse
        // map: for each (row, compactCol, idx), keep the originalCol via compactIndex inverse.
        // The simplest is: for each compactCol, find the originalCol such that compactIndex[originalCol] == compactCol.
        // We'll build a compact→original map once.
        for (int i = 0; i < totalVars; i++)
            if (compactIndex[i] >= 0) compactToOriginal[compactIndex[i]] = i;

        var rowToEntries = new Dictionary<int, List<(int origCol, int idx)>>();
        for (int i = 0; i < structRows.Length; i++)
        {
            if (!rowToEntries.ContainsKey(structRows[i]))
                rowToEntries[structRows[i]] = new List<(int, int)>();
            rowToEntries[structRows[i]].Add((compactToOriginal[structCols[i]], i));
        }

        // The reduced path produces a row already in compact column space, so it indexes by
        // structCols directly rather than through the compact→original detour above.
        var rowToCompactEntries = new Dictionary<int, List<(int compactCol, int idx)>>();
        var reducedRow = reduced is null ? [] : new double[compactToOriginal.Length];
        if (reduced is not null)
            for (int i = 0; i < structRows.Length; i++)
            {
                if (!rowToCompactEntries.ContainsKey(structRows[i]))
                    rowToCompactEntries[structRows[i]] = [];
                rowToCompactEntries[structRows[i]].Add((structCols[i], i));
            }

        return (int n, double* pX, bool newX, int m, int neleJac, int* iRow, int* jCol, double* pValues, nint userData) =>
        {
            if (pValues == null)
            {
                for (int i = 0; i < structRows.Length; i++)
                {
                    iRow[i] = structRows[i];
                    jCol[i] = structCols[i];
                }
            }
            else
            {
                var x = new ReadOnlySpan<double>(pX, n);
                if (newX) syncScratch(x);
                var values = new Span<double>(pValues, neleJac);
                Span<double> gradSpan = grad;

                values.Clear();

                for (int row = 0; row < m; row++)
                {
                    if (reduced is not null)
                    {
                        var reducedSpan = reducedRow.AsSpan(0, n);
                        reducedSpan.Clear();
                        reduced.Gradient(constraints[row].Expression, reducedSpan);
                        foreach (var (compactCol, idx) in rowToCompactEntries[row])
                        {
                            values[idx] = reducedSpan[compactCol];
                            if (!IsValidNumber(values[idx]))
                                return false;
                        }
                        continue;
                    }

                    constraints[row].Expression.AccumulateGradient(scratch, gradSpan);

                    foreach (var (origCol, idx) in rowToEntries[row])
                    {
                        values[idx] = gradSpan[origCol];
                        if (!IsValidNumber(values[idx]))
                            return false;
                        gradSpan[origCol] = 0;  // Clear the sparse entries we used
                    }
                }
            }
            return true;
        };
    }

    private static unsafe EvalHCallback CreateEvalHCallback(int[] structRowsOrig, int[] structColsOrig, int[] structRowsCompact, int[] structColsCompact, Expr objective, Constraint[] constraints, double[] scratch, Action<ReadOnlySpan<double>> syncScratch, int totalVars, ReducedDerivatives? reduced)
    {
        // HessianAccumulator's CSR is indexed in ORIGINAL Variable.Index space — that's what every
        // ExprNode.AccumulateHessian writes into via hess.Add(orig_i, orig_j, value). The compact
        // iRow/jCol vector is what IPOPT consumes, but the values array is identical (entries are
        // ordered the same way).
        var hess = new SparseHessianAccumulator(totalVars, structRowsOrig, structColsOrig);

        return (int hN, double* pX, bool newX, double objFactor, int m, double* lambda, bool newLambda,
                int neleHess, int* iRow, int* jCol, double* pValues, nint userData) =>
        {
            if (pValues == null)
            {
                for (int i = 0; i < structRowsCompact.Length; i++)
                {
                    iRow[i] = structRowsCompact[i];
                    jCol[i] = structColsCompact[i];
                }
            }
            else
            {
                var x = new ReadOnlySpan<double>(pX, hN);
                if (newX) syncScratch(x);
                hess.Clear();

                if (reduced is not null)
                {
                    // [I; X]ᵀ (∇²L + Σ μ_l ∇²E_l) [I; X], accumulated in original-index space like
                    // the direct path — the values array is the same either way.
                    reduced.LagrangianHessian(objFactor, new ReadOnlySpan<double>(lambda, m), hess);
                }
                else
                {
                    objective.AccumulateHessian(scratch, hess, objFactor);
                    for (int row = 0; row < m; row++)
                        constraints[row].Expression.AccumulateHessian(scratch, hess, lambda[row]);
                }

                var values = new Span<double>(pValues, neleHess);
                hess.Values.CopyTo(values);
                for (int i = 0; i < values.Length; i++)
                    if (!IsValidNumber(values[i]))
                        return false;
            }
            return true;
        };
    }

    /// <summary>Folds one partition's statistics into the running aggregate. Norms and violations
    /// take the worst value — the model is only as feasible as its worst sub-problem — step sizes the
    /// smallest, and counts sum. Used both for what the callback reports mid-solve and for the final
    /// <see cref="ModelResult.Statistics"/>, so the two cannot come to mean different things.</summary>
    private static SolveStatistics MergeStatistics(SolveStatistics? completed, SolveStatistics next)
    {
        if (completed is null) return next;   // the min-fields have no neutral element; start from one
        return new SolveStatistics(
            AlgorithmMode: completed.AlgorithmMode == AlgorithmMode.RestorationPhaseMode
                        || next.AlgorithmMode == AlgorithmMode.RestorationPhaseMode
                ? AlgorithmMode.RestorationPhaseMode : AlgorithmMode.RegularMode,
            IterationCount: completed.IterationCount + next.IterationCount,
            ObjectiveValue: completed.ObjectiveValue + next.ObjectiveValue,
            PrimalInfeasibility: Math.Max(completed.PrimalInfeasibility, next.PrimalInfeasibility),
            DualInfeasibility: Math.Max(completed.DualInfeasibility, next.DualInfeasibility),
            ComplementarityMeasure: Math.Max(completed.ComplementarityMeasure, next.ComplementarityMeasure),
            DNorm: Math.Max(completed.DNorm, next.DNorm),
            RegularizationSize: Math.Max(completed.RegularizationSize, next.RegularizationSize),
            DualStepSize: Math.Min(completed.DualStepSize, next.DualStepSize),
            PrimalStepSize: Math.Min(completed.PrimalStepSize, next.PrimalStepSize),
            LineSearchTrials: Math.Max(completed.LineSearchTrials, next.LineSearchTrials));
    }

    /// <summary>How bad a status is, for picking the aggregate. Lower is better.</summary>
    private static int StatusRank(ApplicationReturnStatus status) => status switch
    {
        ApplicationReturnStatus.SolveSucceeded => 0,
        ApplicationReturnStatus.SolvedToAcceptableLevel => 1,
        ApplicationReturnStatus.FeasiblePointFound => 2,
        _ => 3
    };

    /// <summary>Folds the per-partition results into one that reads as if the model had been solved
    /// in a single IPOPT run.</summary>
    private static ModelResult CombineResults(List<ModelResult> results, List<SolvePlan> plans,
        SolveBuffers buffers, double objectiveConstant, Dictionary<Variable, double> fixedSolution)
    {
        // Worst status wins, ties broken by lowest partition index — so the caller cannot mistake a
        // partially-failed solve for a successful one.
        var worst = results[0];
        foreach (var r in results)
            if (StatusRank(r.Status) > StatusRank(worst.Status))
                worst = r;

        // A partly-optimal primal vector is a footgun, so one null solution nulls the aggregate.
        // The per-partition dictionaries stay available on Partitions for debugging.
        Dictionary<Variable, double>? solution = null;
        if (results.All(r => r.Solution is not null))
        {
            solution = new Dictionary<Variable, double>(fixedSolution);
            foreach (var r in results)
                foreach (var (v, value) in r.Solution!)
                    solution[v] = value;
            // A stop request can leave later partitions unsolved. Report their variables at the
            // start point they were seeded with, so the caller still gets a complete vector.
            for (int p = results.Count; p < plans.Count; p++)
            {
                foreach (var v in plans[p].ActiveVariables)
                    solution[v] = buffers.Scratch[v.Index] * v.Scale;
                foreach (var v in plans[p].EliminatedVariables)
                    solution[v] = buffers.Scratch[v.Index] * v.Scale;
            }
        }

        SolveStatistics? folded = null;
        foreach (var r in results)
            folded = MergeStatistics(folded, r.Statistics);
        var statistics = folded! with { ObjectiveValue = folded.ObjectiveValue + objectiveConstant };

        // Partitions are independent, so the model's best iterate is simply each partition's best
        // taken together. Only meaningful when every partition contributed one — a missing snapshot
        // (or a partition an early stop never reached) leaves no defensible value for its variables.
        IterateSnapshot? bestIterate = null;
        if (results.Count == plans.Count && results.All(r => r.BestIterate is not null))
        {
            var merged = new Dictionary<Variable, double>(fixedSolution);
            foreach (var r in results)
                foreach (var (v, value) in r.BestIterate!.Solution)
                    merged[v] = value;
            bestIterate = new IterateSnapshot(
                merged,
                results.Sum(r => r.BestIterate!.ObjectiveValue) + objectiveConstant,
                results.Max(r => r.BestIterate!.PrimalInfeasibility),
                results.All(r => r.BestIterate!.IsFeasible),
                results.Sum(r => r.BestIterate!.IterationCount));
        }

        return new ModelResult(
            worst.Status,
            solution,
            results.Sum(r => r.ObjectiveValue) + objectiveConstant,
            statistics)
        {
            Partitions = results,
            BestIterate = bestIterate,
        };
    }

    private static unsafe EvalHCallback CreateDummyEvalHCallback()
    {
        return (int n, double* pX, bool newX, double objFactor, int m, double* lambda, bool newLambda,
                int neleHess, int* iRow, int* jCol, double* pValues, nint userData) => false;
    }

    public void Dispose()
    {
        _disposed = true;
    }
}

/// <summary>One iterate IPOPT passed through, captured because it was the best seen at the time.
/// See <see cref="ModelResult.BestIterate"/>.</summary>
/// <param name="Solution">Every variable's value at this iterate, eliminated ones included.</param>
/// <param name="ObjectiveValue">The objective there. Under partitioning this is the model-level
/// value, summed across partitions with the objective constant added once.</param>
/// <param name="PrimalInfeasibility">Largest constraint violation there; 0 when unconstrained.</param>
/// <param name="IsFeasible">Whether <paramref name="PrimalInfeasibility"/> was within
/// <see cref="IpoptOptions.ConstraintViolationTolerance"/>. When false, no feasible iterate was ever
/// seen and this is merely the least-infeasible one — treat the objective with suspicion.</param>
/// <param name="IterationCount">The iteration it was found at. Under partitioning, the sum across
/// partitions, matching how <see cref="SolveStatistics.IterationCount"/> aggregates.</param>
public sealed record IterateSnapshot(
    IReadOnlyDictionary<Variable, double> Solution,
    double ObjectiveValue,
    double PrimalInfeasibility,
    bool IsFeasible,
    int IterationCount);

public sealed record ModelResult(
    ApplicationReturnStatus Status,
    IReadOnlyDictionary<Variable, double>? Solution,
    double ObjectiveValue,
    SolveStatistics Statistics)
{
    /// <summary>The individual sub-problem results, smallest sub-problem first, when
    /// <see cref="Model.EnablePartitioning"/> was set and the model decomposed into more than one
    /// partition; empty otherwise. The members above are the model-level aggregate over these.
    /// Covers the partitions that were actually solved, so it excludes the inert partition (which
    /// never reaches IPOPT) and is cut short when a callback requested an early stop.</summary>
    public IReadOnlyList<ModelResult> Partitions { get; init; } = [];

    /// <summary>The best iterate IPOPT passed through, which is not necessarily the one in
    /// <see cref="Solution"/>: a run ending on MaximumIterationsExceeded, RestorationFailed or a
    /// caller stop can finish at a worse point than it visited earlier. "Best" is feasibility-first
    /// — the lowest-objective iterate within
    /// <see cref="IpoptOptions.ConstraintViolationTolerance"/>, falling back to the least-infeasible
    /// one (flagged by <see cref="IterateSnapshot.IsFeasible"/>) when nothing feasible was seen.
    /// Restoration-phase iterates are excluded, their objective belonging to a different NLP.
    ///
    /// Note what this does not promise: the snapshot can be marginally LESS feasible than
    /// <see cref="Solution"/>. Once two points are both inside the violation tolerance they count as
    /// equally feasible and the lower objective wins, so a converged run can report a snapshot
    /// sitting at the edge of the tolerance with an objective a hair under the true optimum. Tighten
    /// <see cref="IpoptOptions.ConstraintViolationTolerance"/> if that matters to you.
    ///
    /// Null when no iterate could be captured — a solve that never reached an iteration, or one
    /// whose iterates were all in restoration. On the model-level aggregate it is also null when any
    /// partition lacks one, including partitions skipped by an early stop; each partition's own
    /// snapshot remains on <see cref="Partitions"/> regardless.</summary>
    public IterateSnapshot? BestIterate { get; init; }
}
