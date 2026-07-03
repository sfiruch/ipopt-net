namespace IpoptNet.Modelling;

internal sealed class SinNode : ExprNode
{
    public ExprNode Argument { get; set; }
    private double[]? _gradBuffer;
    private bool _argIsLinear;
    private int[]? _hessSlots;                    // lower-triangle slots for the gradient outer product
    private HessianAccumulator? _hessSlotsOwner;  // accumulator instance _hessSlots was resolved against

    public SinNode(ExprNode argument) => Argument = argument;

    internal override double EvaluateCore(ReadOnlySpan<double> x) => Math.Sin(Argument.Evaluate(x));

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateGradientCompact(x, compactGrad, multiplier * Math.Cos(arg), sortedVarIndices);
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        // A linear argument has a zero Hessian — skip the subtree walk. (_argIsLinear is false for
        // arguments containing block-eliminated variables, whose Hessian is non-trivial.)
        if (!_argIsLinear)
            Argument.AccumulateHessian(x, hess, multiplier * Math.Cos(arg));

        var coeff = multiplier * -Math.Sin(arg);

        if (coeff == 0.0)
            return;

        Array.Clear(_gradBuffer!);
        Argument.AccumulateGradientCompact(x, _gradBuffer!, 1.0, Argument._sortedVarIndices!);
        AddGradientOuterProduct(hess, coeff, _gradBuffer!, Argument._sortedVarIndices!, ref _hessSlots, ref _hessSlotsOwner);
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
        Argument.Prepare(_model);
        _gradBuffer = new double[Argument._cachedVariables!.Count];
        _argIsLinear = Argument.IsLinear();
    }

    internal override void ClearChildren()
    {
        Argument.Clear();
        _gradBuffer = null;
        _hessSlots = null;
        _hessSlotsOwner = null;
    }
}
