namespace IpoptNet.Modelling;

internal abstract class ExprNode
{
    internal HashSet<Variable>? _cachedVariables;
    internal int[]? _sortedVarIndices;

    internal abstract double Evaluate(ReadOnlySpan<double> x);
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

    internal void Prepare()
    {
        if (_cachedVariables is not null)
            return;

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
        ClearChildren();
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
