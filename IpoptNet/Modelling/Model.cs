using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace IpoptNet.Modelling;

[DebuggerDisplay("Variables: {_variables.Count}, Constraints: {_constraints.Count}")]
public sealed class Model : IDisposable
{
    private readonly List<Variable> _variables = new();
    private readonly List<Constraint> _constraints = new();
    private readonly List<ImplicitBlock> _implicitBlocks = new();
    private Expr? _objective;
    private bool _disposed;

    /// <summary>
    /// IPOPT solver options. Configure before calling Solve().
    /// </summary>
    public IpoptOptions Options { get; } = new();

    /// <summary>
    /// Optional callback invoked at each IPOPT iteration.
    /// Return true to continue, false to request early termination.
    /// </summary>
    public Func<SolveStatistics, bool>? IntermediateCallback { get; set; }

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
    /// are recomputed numerically each evaluation pass via LU on a small dense matrix, and
    /// gradients propagate through the implicit-function theorem.
    ///
    /// Constraints must be equalities (LowerBound == UpperBound) and must be linear in the
    /// eliminated variables (they may be arbitrary in non-eliminated vars / parameters).
    /// Eliminated variables must have infinite bounds and unit Scale.
    ///
    /// When any implicit block is added, the model forces HessianApproximation = LimitedMemory
    /// (exact Hessian propagation through the implicit chain is not yet implemented).
    ///
    /// Blocks must be added in topological order: a block's residuals may reference variables
    /// from previously-added blocks but not from later ones.
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
            if (v.Scale != 1.0)
                throw new ArgumentException(
                    $"Variable x[{v.Index}] has Scale={v.Scale}. Eliminated variables must have unit Scale.");
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

        var varArr = variables.ToArray();
        var resArr = linearEqualities.Select(c => c.Expression).ToArray();
        var block = new ImplicitBlock(varArr, resArr);
        for (int j = 0; j < varArr.Length; j++)
        {
            varArr[j].Block = block;
            varArr[j].IndexInBlock = j;
        }
        _implicitBlocks.Add(block);
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

        return sb.ToString();
    }

    public ModelResult Solve(bool updateStartValues = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_objective is null)
            throw new InvalidOperationException("No objective function set");

        // Exact Hessian propagation through implicit blocks is implemented via VariableNode
        // (PropagateHessian) plus the QuadExprNode cross-product term. Caller chooses Hessian mode.

        // Build compact ↔ original index maps. Eliminated variables get compact index -1 and are
        // not exposed to IPOPT.
        int totalVars = _variables.Count;
        var compactIndex = new int[totalVars];
        var activeVars = new List<Variable>(totalVars);
        for (int i = 0; i < totalVars; i++)
        {
            if (_variables[i].IsEliminated)
                compactIndex[i] = -1;
            else
            {
                compactIndex[i] = activeVars.Count;
                activeVars.Add(_variables[i]);
            }
        }
        int n = activeVars.Count;
        int m = _constraints.Count;

        const double Infinity = 1e19;

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
            gL[i] = _constraints[i].LowerBound;
            gU[i] = _constraints[i].UpperBound;
        }

        // Prepare residual expressions inside each block (raw mode), then objective + constraints
        // (redirect mode — eliminated vars contribute their block's transitive inputs).
        foreach (var block in _implicitBlocks)
            block.PrepareResiduals();
        _objective.Prepare();
        foreach (var constraint in _constraints)
            constraint.Expression.Prepare();

        // Per-evaluation scratch buffer (size = total vars, indexed by Variable.Index).
        // VariableNode.Evaluate reads from this; the model populates it from the IPOPT compact
        // vector before each evaluation pass and then runs every implicit block's Solve in order.
        // For non-eliminated vars, scratch[Index] holds the IPOPT-internal (Scale-divided) value.
        // For eliminated vars (Scale==1 mandated), scratch[Index] holds physical-unit v*.
        var scratch = new double[totalVars];
        // Buffer for block.Solve to extract A rows via AccumulateGradient.
        var blockGradBuffer = new double[totalVars];
        // Per-pass generation counter so a block solves at most once per fresh evaluation.
        long evalGeneration = 0;

        // Verify each block's residuals are linear in their own eliminated vars (fail fast on
        // misuse). Cheap: a couple of extra AccumulateGradient calls per residual at solve start.
        foreach (var block in _implicitBlocks)
            block.VerifyLinearity(totalVars);

        // Helper: synchronize scratch with the IPOPT compact x and run all blocks.
        void SyncScratch(ReadOnlySpan<double> compactX)
        {
            evalGeneration++;
            for (int i = 0; i < n; i++)
                scratch[activeVars[i].Index] = compactX[i];
            foreach (var block in _implicitBlocks)
                block.Solve(scratch, evalGeneration, blockGradBuffer);
        }

        // Analyze sparsity (in compact column space)
        var (jacRows, jacCols) = AnalyzeJacobianSparsity(compactIndex);

        var useLimitedMemory = Options.HessianApproximation == HessianApproximation.LimitedMemory;
        var (hessRowsOrig, hessColsOrig) = useLimitedMemory ? (Array.Empty<int>(), Array.Empty<int>()) : AnalyzeHessianSparsity();
        // IPOPT-facing iRow/jCol must be in compact column space (active-variable indexing).
        var hessRows = new int[hessRowsOrig.Length];
        var hessCols = new int[hessColsOrig.Length];
        for (int i = 0; i < hessRowsOrig.Length; i++)
        {
            hessRows[i] = compactIndex[hessRowsOrig[i]];
            hessCols[i] = compactIndex[hessColsOrig[i]];
            if (hessRows[i] < 0 || hessCols[i] < 0)
                throw new InvalidOperationException(
                    $"Hessian sparsity entry ({hessRowsOrig[i]}, {hessColsOrig[i]}) references an eliminated variable. " +
                    "CollectHessianSparsity must only return non-eliminated variable indices.");
        }

        // Create callbacks
        var evalF = CreateEvalFCallback(scratch, SyncScratch);
        var evalGradF = CreateEvalGradFCallback(scratch, SyncScratch, totalVars, compactIndex, n);
        var evalG = CreateEvalGCallback(scratch, SyncScratch);
        var evalJacG = CreateEvalJacGCallback(jacRows, jacCols, scratch, SyncScratch, totalVars, compactIndex);
        var evalH = useLimitedMemory ? CreateDummyEvalHCallback() : CreateEvalHCallback(hessRowsOrig, hessColsOrig, hessRows, hessCols, scratch, SyncScratch, totalVars);

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

        solver.IntermediateCallback = IntermediateCallback;

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

        // Auto-enable warm start if we have non-zero dual values and user hasn't explicitly set it
        if (Options.WarmStartInitPoint is null &&
            (activeVars.Any(v => v.LowerBoundDualStart != 0 || v.UpperBoundDualStart != 0) ||
             _constraints.Any(c => c.DualStart != 0)))
        {
            if (!solver.SetOption("warm_start_init_point", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'warm_start_init_point' = 'yes'.");
        }

        // Auto-enable grad_f_constant if objective has constant gradients and user hasn't explicitly set it
        if (Options.GradFConstant is null && _objective.IsLinear())
            if (!solver.SetOption("grad_f_constant", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'grad_f_constant' = 'yes'.");

        // Auto-enable jac_c_constant if all equality constraints have constant Jacobians.
        // Note: when implicit blocks are present, "linear" via VariableNode for an eliminated var
        // returns false (since v* depends nonlinearly on inputs). So this is automatically skipped.
        var equalityConstraints = _constraints.Where(c => Math.Abs(c.LowerBound - c.UpperBound) < 1e-15).ToList();
        if (Options.JacCConstant is null && equalityConstraints.All(c => c.Expression.IsLinear()))
            if (!solver.SetOption("jac_c_constant", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'jac_c_constant' = 'yes'.");

        var inequalityConstraints = _constraints.Where(c => Math.Abs(c.LowerBound - c.UpperBound) >= 1e-15).ToList();
        if (Options.JacDConstant is null && inequalityConstraints.All(c => c.Expression.IsLinear()))
            if (!solver.SetOption("jac_d_constant", "yes"))
                throw new InvalidOperationException("IPOPT rejected option 'jac_d_constant' = 'yes'.");

        if (Options.HessianConstant is null && !useLimitedMemory &&
            _objective.IsAtMostQuadratic() && _constraints.All(c => c.Expression.IsLinear()))
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
            constraintMultipliers[i] = _constraints[i].DualStart;

        for (int i = 0; i < n; i++)
        {
            lowerBoundMultipliers[i] = activeVars[i].LowerBoundDualStart;
            upperBoundMultipliers[i] = activeVars[i].UpperBoundDualStart;
        }

        status = solver.Solve(x, out objValue, out statistics, constraintValues, constraintMultipliers,
                                  lowerBoundMultipliers, upperBoundMultipliers);

        // Build solution. We expose all variables (including eliminated ones) in the dictionary
        // so callers see consistent values; eliminated vars are read from scratch after a final
        // sync at the returned x.
        var solution = (Dictionary<Variable, double>?)null;


            solution = new Dictionary<Variable, double>();
            // Sync once more to populate scratch with eliminated values at the final iterate.
            SyncScratch(x.AsSpan());
            for (int i = 0; i < n; i++)
                solution[activeVars[i]] = Math.Clamp(x[i], xL[i], xU[i]) * activeVars[i].Scale;
            foreach (var v in _variables)
                if (v.IsEliminated)
                    solution[v] = scratch[v.Index];

        // Update variable Start values and dual variables if requested and solution is usable
        if (updateStartValues && status is
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
            foreach (var v in _variables)
                if (v.IsEliminated)
                    v.Start = solution[v];

            for (int i = 0; i < m; i++)
                _constraints[i].DualStart = constraintMultipliers[i];
        }

        // Clear cached variables to free memory after optimization
        _objective.Clear();
        foreach (var constraint in _constraints)
            constraint.Expression.Clear();
        foreach (var block in _implicitBlocks)
            block.ClearResiduals();

        return new ModelResult(status, solution, objValue, statistics);
    }

    private (int[] rows, int[] cols) AnalyzeJacobianSparsity(int[] compactIndex)
    {
        var entries = new HashSet<(int row, int col)>();
        var vars = new HashSet<Variable>();

        for (int i = 0; i < _constraints.Count; i++)
        {
            vars.Clear();
            _constraints[i].Expression.CollectVariables(vars);
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
    private (int[] rows, int[] cols) AnalyzeHessianSparsity()
    {
        var entries = new HashSet<(int row, int col)>();
        _objective?.CollectHessianSparsity(entries);
        foreach (var c in _constraints)
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

    private unsafe EvalFCallback CreateEvalFCallback(double[] scratch, Action<ReadOnlySpan<double>> syncScratch)
    {
        return (int n, double* pX, bool newX, double* objValue, nint userData) =>
        {
            var x = new ReadOnlySpan<double>(pX, n);
            if (newX) syncScratch(x);
            *objValue = _objective!.Evaluate(scratch);
            return IsValidNumber(*objValue);
        };
    }

    private unsafe EvalGradFCallback CreateEvalGradFCallback(double[] scratch, Action<ReadOnlySpan<double>> syncScratch, int totalVars, int[] compactIndex, int activeCount)
    {
        var fullGrad = new double[totalVars];
        return (int n, double* pX, bool newX, double* pGradF, nint userData) =>
        {
            var x = new ReadOnlySpan<double>(pX, n);
            if (newX) syncScratch(x);
            var gradF = new Span<double>(pGradF, n);
            gradF.Clear();
            // The objective's _cachedVariables only includes non-eliminated vars (CollectVariables
            // in redirect mode walks blocks). AccumulateGradient writes into fullGrad indexed by
            // Variable.Index. We re-pack into the IPOPT compact gradF using compactIndex.
            Array.Clear(fullGrad);
            _objective!.AccumulateGradient(scratch, fullGrad);
            for (int i = 0; i < totalVars; i++)
            {
                int ci = compactIndex[i];
                if (ci < 0) continue;
                gradF[ci] = fullGrad[i];
            }

            for (int i = 0; i < n; i++)
                if (!IsValidNumber(gradF[i]))
                    return false;

            return true;
        };
    }

    public static bool IsValidNumber(double v) => !double.IsInfinity(v) && !double.IsNaN(v);

    private unsafe EvalGCallback CreateEvalGCallback(double[] scratch, Action<ReadOnlySpan<double>> syncScratch)
    {
        return (int n, double* pX, bool newX, int m, double* pG, nint userData) =>
        {
            var x = new ReadOnlySpan<double>(pX, n);
            if (newX) syncScratch(x);
            var g = new Span<double>(pG, m);
            for (int i = 0; i < m; i++)
            {
                g[i] = _constraints[i].Expression.Evaluate(scratch);
                if (!IsValidNumber(g[i]))
                    return false;
            }
            return true;
        };
    }

    private unsafe EvalJacGCallback CreateEvalJacGCallback(int[] structRows, int[] structCols, double[] scratch, Action<ReadOnlySpan<double>> syncScratch, int totalVars, int[] compactIndex)
    {
        // structCols are in COMPACT column space. To map a constraint's gradient (computed in
        // original-variable-index space) into the right value slots, we precompute an inverse
        // map: for each (row, compactCol, idx), keep the originalCol via compactIndex inverse.
        // The simplest is: for each compactCol, find the originalCol such that compactIndex[originalCol] == compactCol.
        // We'll build a compact→original map once.
        var compactToOriginal = new int[totalVars];  // upper bound; only first activeCount slots used
        for (int i = 0; i < totalVars; i++)
            if (compactIndex[i] >= 0) compactToOriginal[compactIndex[i]] = i;

        var rowToEntries = new Dictionary<int, List<(int origCol, int idx)>>();
        for (int i = 0; i < structRows.Length; i++)
        {
            if (!rowToEntries.ContainsKey(structRows[i]))
                rowToEntries[structRows[i]] = new List<(int, int)>();
            rowToEntries[structRows[i]].Add((compactToOriginal[structCols[i]], i));
        }

        // Allocate gradient buffer once and reuse it
        var grad = new double[totalVars];

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
                    _constraints[row].Expression.AccumulateGradient(scratch, gradSpan);

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

    private unsafe EvalHCallback CreateEvalHCallback(int[] structRowsOrig, int[] structColsOrig, int[] structRowsCompact, int[] structColsCompact, double[] scratch, Action<ReadOnlySpan<double>> syncScratch, int totalVars)
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

                _objective!.AccumulateHessian(scratch, hess, objFactor);
                for (int row = 0; row < m; row++)
                    _constraints[row].Expression.AccumulateHessian(scratch, hess, lambda[row]);

                var values = new Span<double>(pValues, neleHess);
                hess.Values.CopyTo(values);
                for (int i = 0; i < values.Length; i++)
                    if (!IsValidNumber(values[i]))
                        return false;
            }
            return true;
        };
    }

    private unsafe EvalHCallback CreateDummyEvalHCallback()
    {
        return (int n, double* pX, bool newX, double objFactor, int m, double* lambda, bool newLambda,
                int neleHess, int* iRow, int* jCol, double* pValues, nint userData) => false;
    }

    public void Dispose()
    {
        _disposed = true;
    }
}

public sealed record ModelResult(
    ApplicationReturnStatus Status,
    IReadOnlyDictionary<Variable, double>? Solution,
    double ObjectiveValue,
    SolveStatistics Statistics);
