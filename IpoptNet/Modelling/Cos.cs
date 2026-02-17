namespace IpoptNet.Modelling;

internal sealed class CosNode : ExprNode
{
    public ExprNode Argument { get; set; }
    private double[]? _gradBuffer;

    public CosNode(ExprNode argument) => Argument = argument;

    internal override double Evaluate(ReadOnlySpan<double> x) => Math.Cos(Argument.Evaluate(x));

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateGradientCompact(x, compactGrad, multiplier * -Math.Sin(arg), sortedVarIndices);
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, hess, multiplier * -Math.Sin(arg));

        var coeff = multiplier * -Math.Cos(arg);

        Array.Clear(_gradBuffer!);
        Argument.AccumulateGradientCompact(x, _gradBuffer!, 1.0, Argument._sortedVarIndices!);

        var sorted = Argument._sortedVarIndices!;
        for (int i = 0; i < sorted.Length; i++)
        {
            var gI = _gradBuffer![i];
            for (int j = 0; j <= i; j++)
                hess.Add(sorted[i], sorted[j], coeff * gI * _gradBuffer[j]);
        }
    }

    internal override void CollectVariables(HashSet<Variable> variables) => Argument.CollectVariables(variables);
    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries)
    {
        if (!Argument.IsConstantWrtX())
        {
            AddClique(entries, Argument._cachedVariables!);
        }
    }
    internal override bool IsConstantWrtX() => Argument.IsConstantWrtX();
    internal override bool IsLinear() => Argument.IsConstantWrtX();
    internal override bool IsAtMostQuadratic() => Argument.IsConstantWrtX();

    internal override void PrepareChildren()
    {
        Argument.Prepare();
        _gradBuffer = new double[Argument._cachedVariables!.Count];
    }

    internal override void ClearChildren()
    {
        Argument.Clear();
        _gradBuffer = null;
    }
}
