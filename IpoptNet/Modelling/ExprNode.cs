namespace IpoptNet.Modelling;

internal abstract class ExprNode
{
    internal HashSet<Variable>? _cachedVariables;
    internal int[]? _sortedVarIndices;

    /// <summary>Model this node was prepared for, set via <see cref="Prepare"/>. Enables the
    /// per-pass value cache below; null (unprepared, or prepared without a model) disables it.</summary>
    internal Model? _model;
    private double _cachedValue;
    private long _cachedValueGeneration; // matches Model.EvalGeneration when _cachedValue is valid; 0 = invalid

    /// <summary>
    /// Evaluates this node, memoizing the result for the current evaluation pass. Gradient and
    /// Hessian passes re-evaluate subtrees they differentiate (e.g. PowerOp evaluates its base,
    /// Product evaluates every factor); the cache collapses those repeats to O(1) per node.
    /// Caching is keyed on <see cref="Model.EvalGeneration"/>, which the model bumps whenever the
    /// evaluation buffer's contents change (SyncScratch, implicit-block solves, ...), and is only
    /// active inside <see cref="Model.Solve"/> (<see cref="Model.ValueCachingActive"/>) so ad-hoc
    /// Evaluate calls on arbitrary x vectors outside a solve are never served stale values.
    /// </summary>
    internal double Evaluate(ReadOnlySpan<double> x)
    {
        var m = _model;
        if (m is null || !m.ValueCachingActive)
            return EvaluateCore(x);
        if (_cachedValueGeneration == m.EvalGeneration)
            return _cachedValue;
        var v = EvaluateCore(x);
        _cachedValueGeneration = m.EvalGeneration;
        _cachedValue = v;
        return v;
    }

    internal abstract double EvaluateCore(ReadOnlySpan<double> x);
    internal abstract void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices);
    internal abstract void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier);
    internal abstract void CollectVariables(HashSet<Variable> variables);
    internal abstract void CollectHessianSparsity(HashSet<(int row, int col)> entries);
    internal abstract bool IsConstantWrtX();
    internal abstract bool IsLinear();
    internal abstract bool IsAtMostQuadratic();

    internal virtual void PrepareChildren() { }
    internal virtual void ClearChildren() { }
    internal virtual bool IsSimpleForPrinting() => false;

    internal void Prepare(Model? model = null)
    {
        if (_cachedVariables is not null)
            return;

        _model = model;
        _cachedValueGeneration = 0;

        _cachedVariables = new HashSet<Variable>();
        CollectVariables(_cachedVariables);

        // Build sorted variable indices
        _sortedVarIndices = new int[_cachedVariables.Count];
        {
            var i = 0;
            foreach (var v in _cachedVariables)
                _sortedVarIndices[i++] = v.Index;
        }
        Array.Sort(_sortedVarIndices);

        // Recursively cache for children
        PrepareChildren();
    }

    internal void Clear()
    {
        _cachedVariables = null;
        _sortedVarIndices = null;
        _model = null;
        _cachedValueGeneration = 0;
        ClearChildren();
    }

    /// <summary>Adds coeff · ∇child ⊗ ∇child (lower triangle over <paramref name="sorted"/>) to
    /// <paramref name="hess"/>. The clique's accumulator slots are resolved once per accumulator
    /// instance and cached in the caller's <paramref name="slots"/>/<paramref name="slotsOwner"/>
    /// fields, replacing a CSR binary search per entry with a direct indexed add.</summary>
    private protected static void AddGradientOuterProduct(HessianAccumulator hess, double coeff, double[] grad, int[] sorted,
        ref int[]? slots, ref HessianAccumulator? slotsOwner)
    {
        if (!ReferenceEquals(slotsOwner, hess))
        {
            slots ??= new int[sorted.Length * (sorted.Length + 1) / 2];
            var t = 0;
            for (int i = 0; i < sorted.Length; i++)
                for (int j = 0; j <= i; j++)
                    slots[t++] = hess.GetSlot(sorted[i], sorted[j]);
            slotsOwner = hess;
        }

        var s = 0;
        for (int i = 0; i < sorted.Length; i++)
        {
            var gI = grad[i];
            for (int j = 0; j <= i; j++)
                hess.AddAtSlot(slots![s++], coeff * gI * grad[j]);
        }
    }

    internal static void AddSparsityEntry(HashSet<(int row, int col)> entries, int i, int j)
    {
        if (i < j) (i, j) = (j, i);
        entries.Add((i, j));
    }

    internal static void AddClique(HashSet<(int row, int col)> entries, HashSet<Variable> variables)
    {
        var vars = variables.ToArray();
        for (int i = 0; i < vars.Length; i++)
            for (int j = 0; j <= i; j++)
                AddSparsityEntry(entries, vars[i].Index, vars[j].Index);
    }
}
