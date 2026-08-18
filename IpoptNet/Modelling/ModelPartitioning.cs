using System.Diagnostics;
using System.Text;

namespace IpoptNet.Modelling;

/// <summary>One disconnected sub-problem of a <see cref="Model"/>. Its variables share no
/// constraint, implicit block, or objective term with any other partition, so its optimum is
/// independent of theirs.</summary>
/// <param name="Index">0-based position in <see cref="ModelPartitioning.Partitions"/>.</param>
/// <param name="Variables">Non-eliminated variables, ascending <see cref="Variable.Index"/>.</param>
/// <param name="EliminatedVariables">Implicit-block variables, ascending <see cref="Variable.Index"/>.</param>
/// <param name="Constraints">Constraints, in model registration order.</param>
/// <param name="ImplicitBlockCount">Number of implicit blocks wholly owned by this partition.</param>
/// <param name="ObjectiveTermCount">Number of flattened objective terms assigned to this partition.</param>
/// <param name="IsInert">True when this partition has no constraints, no blocks and no objective
/// terms — the coalesced group of variables the model never actually references. Reported for
/// diagnostics; <see cref="Model.Solve"/> resolves these from their start point instead of solving
/// them, so this partition has no counterpart in <see cref="ModelResult.Partitions"/>.</param>
public sealed record ModelPartition(
    int Index,
    IReadOnlyList<Variable> Variables,
    IReadOnlyList<Variable> EliminatedVariables,
    IReadOnlyList<Constraint> Constraints,
    int ImplicitBlockCount,
    int ObjectiveTermCount,
    bool IsInert);

/// <summary>The decomposition of a <see cref="Model"/> into disconnected sub-problems.
/// See <see cref="Model.AnalyzePartitions"/>.</summary>
public sealed record ModelPartitioning(IReadOnlyList<ModelPartition> Partitions)
{
    /// <summary>True when the model does not decompose. <see cref="Model.Solve"/> then takes the
    /// ordinary single-solve path even with <see cref="Model.EnablePartitioning"/> set.</summary>
    public bool IsTrivial => Partitions.Count <= 1;

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Partitions: {Partitions.Count}");
        foreach (var p in Partitions)
            sb.AppendLine(Describe(p));
        return sb.ToString();
    }

    internal static string Describe(ModelPartition p) => p.IsInert
        ? $"  Partition[{p.Index}]: {p.Variables.Count} var(s) [inert]"
        : $"  Partition[{p.Index}]: {p.Variables.Count} var(s), {p.EliminatedVariables.Count} eliminated, "
          + $"{p.Constraints.Count} constraint(s), {p.ImplicitBlockCount} block(s), {p.ObjectiveTermCount} objective term(s)";
}

/// <summary>Identifies which sub-problem an <see cref="Model.IntermediateCallback"/> invocation
/// belongs to. When partitioning is off or the model does not decompose, this reports
/// <c>Index 0, Count 1</c> and <see cref="LocalStatistics"/> equals the statistics passed
/// alongside it.</summary>
/// <param name="Index">0-based index of the partition currently iterating.</param>
/// <param name="Count">Total number of partitions this solve was split into.</param>
/// <param name="VariableCount">Size of this partition's IPOPT decision vector.</param>
/// <param name="ConstraintCount">Number of constraints in this partition.</param>
/// <param name="LocalStatistics">The raw, partition-local statistics as IPOPT reported them.
/// The <see cref="SolveStatistics"/> passed as the callback's first argument is instead
/// normalised to model-level quantities — see <see cref="Model.IntermediateCallback"/>.</param>
public sealed record PartitionInfo(
    int Index,
    int Count,
    int VariableCount,
    int ConstraintCount,
    SolveStatistics LocalStatistics);

public sealed partial class Model
{
    /// <summary>
    /// On by default. <see cref="Solve"/> first decomposes the model into disconnected sub-problems
    /// (see <see cref="AnalyzePartitions"/>) and solves each one with its own IPOPT instance,
    /// sequentially, in ascending partition order. This is mathematically exact — the sub-problems
    /// share no variable through any constraint, implicit block, or objective term, so their optima
    /// are independent — and typically much cheaper, because IPOPT's linear-algebra cost grows
    /// superlinearly with problem size.
    ///
    /// The observable result is equivalent to an unpartitioned solve: every partition is always
    /// processed (a failing one never suppresses the others), <see cref="ModelResult"/> reports
    /// model-level aggregates, and the statistics handed to <see cref="IntermediateCallback"/> are
    /// normalised to model-level quantities. Per-partition detail is available on
    /// <see cref="ModelResult.Partitions"/> and <see cref="PartitionInfo.LocalStatistics"/>.
    ///
    /// Set it to false to force the single whole-model solve — the pre-partitioning behaviour,
    /// byte-identical, and skipping the decomposition analysis entirely.
    ///
    /// Three things do differ when it is on and the model decomposes.
    /// <see cref="IpoptOptions.MaxIterations"/> applies <em>per partition</em> — it is a
    /// "don't spin forever on this sub-problem" guard, and dividing it would make a later partition
    /// fail merely for having followed a hard one. <see cref="IpoptOptions.MaxWallTime"/> and
    /// <see cref="IpoptOptions.MaxCpuTime"/>, by contrast, stay <em>model-wide deadlines</em>: each
    /// partition is given what remains of the budget, so N partitions cannot take N times as long as
    /// the caller allowed. (Elapsed wall time is measured exactly; elapsed CPU time is taken from the
    /// process total, which over-counts when other threads are busy and therefore errs toward
    /// stopping sooner.) <see cref="IpoptOptions.OutputFile"/> receives the concatenation of N IPOPT
    /// runs; and <em>inert</em> variables are not sent to IPOPT at all.
    ///
    /// An inert variable is one appearing in no constraint, no implicit block and no objective term.
    /// Nothing optimises it and nothing constrains it, so an IPOPT run would burn a full problem
    /// setup only to return wherever the barrier happened to drift it — a value that depends on the
    /// iteration count of the surrounding problem and is therefore arbitrary either way. Such
    /// variables are instead resolved directly from their start point: an explicit
    /// <see cref="Variable.Start"/> clamped to bounds, otherwise the same bound-derived default
    /// IPOPT would have been seeded with. That makes the reported value deterministic and
    /// explainable, at the cost of differing from what an unpartitioned solve happens to return for
    /// them. Their <see cref="Variable.Start"/> and duals are left untouched, and every variable the
    /// model actually references is unaffected.
    ///
    /// Deliberately not on <see cref="IpoptOptions"/>: every member there maps to a native IPOPT
    /// option key, and this is a modelling-layer concern with no IPOPT counterpart.
    /// </summary>
    public bool EnablePartitioning { get; set; } = true;

    /// <summary>
    /// Computes this model's decomposition into disconnected sub-problems without solving.
    /// Two variables land in the same partition when they are transitively related by appearing
    /// together in a constraint, by belonging to or being read by the same implicit block, or by
    /// appearing in the same flattened objective term.
    ///
    /// Pure: does not mutate model state, does not require the expression graph to be prepared, and
    /// is unaffected by <see cref="EnablePartitioning"/> (it reports the decomposition either way). Partitions are ordered by their smallest
    /// <see cref="Variable.Index"/>, so the result is deterministic and reproducible.
    ///
    /// Note that variables the model never references ("inert" — in no constraint, no block and no
    /// objective term) are reported coalesced into a single <see cref="ModelPartition.IsInert"/>
    /// partition rather than as one component each. <see cref="Solve"/> does not hand that partition
    /// to IPOPT — it resolves those variables from their start point — so it contributes no entry to
    /// <see cref="ModelResult.Partitions"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">No objective function has been set.</exception>
    public ModelPartitioning AnalyzePartitions()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_objective is null)
            throw new InvalidOperationException("No objective function set");

        var layout = ComputePartitionLayout();
        var partitions = new ModelPartition[layout.Count];
        for (int p = 0; p < layout.Count; p++)
            partitions[p] = new ModelPartition(
                p,
                layout.ActiveVariables[p],
                layout.EliminatedVariables[p],
                layout.Constraints[p],
                layout.Blocks[p].Length,
                layout.ObjectiveTerms[p].Count,
                layout.IsInert[p]);
        return new ModelPartitioning(partitions);
    }

    // ---------------------------------------------------------------------------------------
    // Objective term enumeration
    //
    // The flattened additive terms of the objective. LinExprNode and QuadExprNode keep their
    // summands in flat lists (Expr.operator + / += merge nested nodes into them, and
    // operator /(Expr, double) folds the divisor into the weights rather than wrapping the node),
    // so an objective built up as `obj += ...; obj = obj / n;` is a single flat node whose terms
    // can be walked directly. Anything else is treated as one opaque term covering every variable
    // the objective touches — which collapses the model to a single partition, the safe answer.
    //
    // Using term variable-sets (rather than deriving coupling from CollectHessianSparsity) is
    // deliberate: the same enumeration that defines the components also defines each partition's
    // objective slice, so the two cannot disagree. It also needs no Prepare(), which is what lets
    // AnalyzePartitions() be a pure, standalone API. Where it differs from the Hessian criterion it
    // only ever over-merges — slower, never wrong.
    // ---------------------------------------------------------------------------------------

    private enum ObjectiveTermKind { Linear, Quadratic, Opaque }

    private readonly record struct ObjectiveTerm(ObjectiveTermKind Kind, int Index);

    private List<ObjectiveTerm> EnumerateObjectiveTerms()
    {
        var terms = new List<ObjectiveTerm>();
        switch (_objective!._node)
        {
            case LinExprNode lin:
                for (int i = 0; i < lin.Terms.Count; i++)
                    terms.Add(new ObjectiveTerm(ObjectiveTermKind.Linear, i));
                break;
            case QuadExprNode quad:
                for (int i = 0; i < quad.LinearTerms.Count; i++)
                    terms.Add(new ObjectiveTerm(ObjectiveTermKind.Linear, i));
                for (int i = 0; i < quad.QuadraticTerms1.Count; i++)
                    terms.Add(new ObjectiveTerm(ObjectiveTermKind.Quadratic, i));
                break;
            default:
                terms.Add(new ObjectiveTerm(ObjectiveTermKind.Opaque, -1));
                break;
        }
        return terms;
    }

    private void CollectObjectiveTermVariables(ObjectiveTerm term, HashSet<Variable> into)
    {
        switch (_objective!._node)
        {
            case LinExprNode lin when term.Kind == ObjectiveTermKind.Linear:
                lin.Terms[term.Index].CollectVariables(into);
                break;
            case QuadExprNode quad when term.Kind == ObjectiveTermKind.Linear:
                quad.LinearTerms[term.Index].CollectVariables(into);
                break;
            case QuadExprNode quad:
                quad.QuadraticTerms1[term.Index].CollectVariables(into);
                quad.QuadraticTerms2[term.Index].CollectVariables(into);
                break;
            default:
                _objective._node.CollectVariables(into);
                break;
        }
    }

    /// <summary>The additive constant carried by a flattened objective node. Held out of every
    /// partition slice and added back exactly once during aggregation, so it cannot be
    /// double-counted. An opaque objective keeps its constant inside the expression itself.</summary>
    private double ObjectiveConstantTerm => _objective!._node switch
    {
        LinExprNode lin => lin.ConstantTerm,
        QuadExprNode quad => quad.ConstantTerm,
        _ => 0.0
    };

    // ---------------------------------------------------------------------------------------
    // Partition analysis
    // ---------------------------------------------------------------------------------------

    /// <summary>Everything the solve driver and the inspection API need, computed once.</summary>
    private sealed class PartitionLayout
    {
        public required int Count;
        public required int[] PartitionOfVariable;          // by Variable.Index
        public required Variable[][] ActiveVariables;
        public required Variable[][] EliminatedVariables;
        public required Constraint[][] Constraints;
        public required ImplicitBlock[][] Blocks;
        public required List<ObjectiveTerm>[] ObjectiveTerms;
        public required bool[] IsInert;
    }

    private static int FindRoot(int[] parent, int i)
    {
        while (parent[i] != i)
            i = parent[i] = parent[parent[i]];   // path halving
        return i;
    }

    private static void Union(int[] parent, int[] size, int a, int b)
    {
        a = FindRoot(parent, a);
        b = FindRoot(parent, b);
        if (a == b) return;
        if (size[a] < size[b]) (a, b) = (b, a);
        parent[b] = a;
        size[a] += size[b];
    }

    private static void UnionAll(int[] parent, int[] size, bool[] referenced, HashSet<Variable> vars)
    {
        int first = -1;
        foreach (var v in vars)
        {
            referenced[v.Index] = true;
            if (first < 0) first = v.Index;
            else Union(parent, size, first, v.Index);
        }
    }

    private PartitionLayout ComputePartitionLayout()
    {
        int totalVars = _variables.Count;
        var parent = new int[totalVars];
        var size = new int[totalVars];
        for (int i = 0; i < totalVars; i++) { parent[i] = i; size[i] = 1; }

        // Variables the model actually references. The rest are "inert" and get coalesced below.
        var referenced = new bool[totalVars];
        var vars = new HashSet<Variable>();
        bool decomposable = true;

        // (1) Implicit blocks are atomic: a block's eliminated variables and everything its
        // residuals read must be solved together. Raw mode makes an eliminated VariableNode report
        // itself, so a chain of blocks merges transitively. This is a third incidence source —
        // AddImplicitBlock removes the block's equalities from _constraints, so step (2) cannot
        // see them.
        foreach (var block in _implicitBlocks)
        {
            vars.Clear();
            using (EnterRawMode())
                foreach (var r in block.Residuals)
                    r.CollectVariables(vars);
            foreach (var v in block.Variables)
                vars.Add(v);
            UnionAll(parent, size, referenced, vars);
        }

        // (2) Constraints, in redirect mode — matching what AnalyzeJacobianSparsity sees.
        foreach (var c in _constraints)
        {
            vars.Clear();
            c.Expression.CollectVariables(vars);
            // A constraint over no variables is a constant assertion. Rather than pick an arbitrary
            // partition to report its (in)feasibility from, decline to decompose at all.
            if (vars.Count == 0) decomposable = false;
            UnionAll(parent, size, referenced, vars);
        }

        // (3) Objective terms.
        var objectiveTerms = EnumerateObjectiveTerms();
        var termFirstVariable = new int[objectiveTerms.Count];
        for (int t = 0; t < objectiveTerms.Count; t++)
        {
            vars.Clear();
            CollectObjectiveTermVariables(objectiveTerms[t], vars);
            UnionAll(parent, size, referenced, vars);
            // A variable-free term is a constant contribution; park it on the first partition so it
            // is evaluated exactly once.
            termFirstVariable[t] = vars.Count == 0 ? -1 : vars.Min(v => v.Index);
        }

        // Variables the model never references would each be their own singleton component. Coalesce
        // them into one partition instead — otherwise a model carrying N idle variables would pay
        // full IPOPT problem-creation cost N times to determine nothing.
        //
        // The group is reported here for diagnostics, but BuildSolvePlans gives it no plan: nothing
        // optimises or constrains an inert variable, so a solve would only return wherever the
        // barrier drifted it. That value is arbitrary either way — it depends on the surrounding
        // problem's iteration count, so it cannot match an unpartitioned solve regardless — and
        // resolving from the start point at least makes it deterministic. See EnablePartitioning.
        int firstInert = -1;
        for (int i = 0; i < totalVars; i++)
        {
            if (referenced[i]) continue;
            if (firstInert < 0) firstInert = i;
            else Union(parent, size, firstInert, i);
        }

        if (!decomposable)
            for (int i = 1; i < totalVars; i++)
                Union(parent, size, 0, i);

        // Group by root, ordered by ascending minimum Variable.Index — deterministic, and never
        // dependent on hash-set iteration order.
        var partitionOfRoot = new Dictionary<int, int>();
        var partitionOfVariable = new int[totalVars];
        for (int i = 0; i < totalVars; i++)
        {
            int root = FindRoot(parent, i);
            if (!partitionOfRoot.TryGetValue(root, out int p))
                partitionOfRoot[root] = p = partitionOfRoot.Count;
            partitionOfVariable[i] = p;
        }
        int count = partitionOfRoot.Count;

        var active = new List<Variable>[count];
        var eliminated = new List<Variable>[count];
        var constraints = new List<Constraint>[count];
        var blocks = new List<ImplicitBlock>[count];
        var terms = new List<ObjectiveTerm>[count];
        var inert = new bool[count];
        for (int p = 0; p < count; p++)
        {
            active[p] = [];
            eliminated[p] = [];
            constraints[p] = [];
            blocks[p] = [];
            terms[p] = [];
            inert[p] = true;
        }

        foreach (var v in _variables)
            (v.IsEliminated ? eliminated : active)[partitionOfVariable[v.Index]].Add(v);

        // Constraints, blocks and objective terms keep their original relative order within a
        // partition. For blocks that is load-bearing: AddImplicitBlock enforces topological order
        // and SyncScratch relies on solving them in it.
        foreach (var c in _constraints)
        {
            vars.Clear();
            c.Expression.CollectVariables(vars);
            int p = partitionOfVariable[vars.Min(v => v.Index)];
            constraints[p].Add(c);
            inert[p] = false;
        }
        foreach (var block in _implicitBlocks)
        {
            int p = partitionOfVariable[block.Variables[0].Index];
            blocks[p].Add(block);
            inert[p] = false;
        }
        for (int t = 0; t < objectiveTerms.Count; t++)
        {
            int p = termFirstVariable[t] < 0 ? 0 : partitionOfVariable[termFirstVariable[t]];
            terms[p].Add(objectiveTerms[t]);
            inert[p] = false;
        }

        Debug.Assert(terms.Sum(t => t.Count) == objectiveTerms.Count,
            "Every objective term must be assigned to exactly one partition.");

        return new PartitionLayout
        {
            Count = count,
            PartitionOfVariable = partitionOfVariable,
            ActiveVariables = [.. active.Select(l => l.ToArray())],
            EliminatedVariables = [.. eliminated.Select(l => l.ToArray())],
            Constraints = [.. constraints.Select(l => l.ToArray())],
            Blocks = [.. blocks.Select(l => l.ToArray())],
            ObjectiveTerms = terms,
            IsInert = inert,
        };
    }

    // ---------------------------------------------------------------------------------------
    // Objective slicing
    // ---------------------------------------------------------------------------------------

    /// <summary>Builds the objective for one partition: a fresh flat node holding only that
    /// partition's terms, with the additive constant stripped (it is added back once during
    /// aggregation). Mirrors how <c>Expr.operator /(Expr, double)</c> already rebuilds these nodes
    /// via object initialisers, bypassing the re-flattening constructors.</summary>
    private Expr BuildPartitionObjective(List<ObjectiveTerm> terms)
    {
        if (terms.Count == 0)
            return new Expr(new ConstantNode(0.0));

        switch (_objective!._node)
        {
            case LinExprNode lin:
            {
                var node = new LinExprNode { Terms = [], Weights = [], ConstantTerm = 0.0 };
                foreach (var t in terms)
                {
                    node.Terms.Add(lin.Terms[t.Index]);
                    node.Weights.Add(lin.Weights[t.Index]);
                }
                return new Expr(node);
            }
            case QuadExprNode quad:
            {
                var node = new QuadExprNode
                {
                    LinearTerms = [], LinearWeights = [],
                    QuadraticTerms1 = [], QuadraticTerms2 = [], QuadraticWeights = [],
                    ConstantTerm = 0.0
                };
                foreach (var t in terms)
                    if (t.Kind == ObjectiveTermKind.Linear)
                    {
                        node.LinearTerms.Add(quad.LinearTerms[t.Index]);
                        node.LinearWeights.Add(quad.LinearWeights[t.Index]);
                    }
                    else
                    {
                        node.QuadraticTerms1.Add(quad.QuadraticTerms1[t.Index]);
                        node.QuadraticTerms2.Add(quad.QuadraticTerms2[t.Index]);
                        node.QuadraticWeights.Add(quad.QuadraticWeights[t.Index]);
                    }
                return new Expr(node);
            }
            default:
                // The single opaque term: this partition owns the whole objective, constant included.
                return _objective;
        }
    }
}
