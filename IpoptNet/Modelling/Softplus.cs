namespace IpoptNet.Modelling;

internal sealed class SoftplusNode : ExprNode
{
    public ExprNode Argument { get; set; }
    public double Sharpness { get; }
    private double[]? _gradBuffer;
    private bool _argIsLinear;
    private int[]? _hessSlots;                    // lower-triangle slots for the gradient outer product
    private HessianAccumulator? _hessSlotsOwner;  // accumulator instance _hessSlots was resolved against

    public SoftplusNode(ExprNode argument, double sharpness)
    {
        Argument = argument;
        Sharpness = sharpness;
    }

    // Numerically stable: (1/k) * (max(t, 0) + ln(1 + exp(-|t|))) where t = k * u
    internal override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var t = Sharpness * Argument.Evaluate(x);
        return (1.0 / Sharpness) * (Math.Max(t, 0) + Math.Log(1.0 + Math.Exp(-Math.Abs(t))));
    }

    // d/du softplus(u) = sigmoid(k * u)
    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        var t = Sharpness * Argument.Evaluate(x);
        var sigma = StableSigmoid(t);
        Argument.AccumulateGradientCompact(x, compactGrad, multiplier * sigma, sortedVarIndices);
    }

    // d²/du² softplus(u) = k * sigmoid(k*u) * (1 - sigmoid(k*u))
    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var t = Sharpness * Argument.Evaluate(x);
        var sigma = StableSigmoid(t);

        // A linear argument has a zero Hessian — skip the subtree walk. (_argIsLinear is false for
        // arguments containing block-eliminated variables, whose Hessian is non-trivial.)
        if (!_argIsLinear)
            Argument.AccumulateHessian(x, hess, multiplier * sigma);

        var coeff = multiplier * Sharpness * sigma * (1.0 - sigma);

        if (coeff == 0.0)
            return;

        Array.Clear(_gradBuffer!);
        Argument.AccumulateGradientCompact(x, _gradBuffer!, 1.0, Argument._sortedVarIndices!);
        AddGradientOuterProduct(hess, coeff, _gradBuffer!, Argument._sortedVarIndices!, ref _hessSlots, ref _hessSlotsOwner);
    }

    private static double StableSigmoid(double t)
    {
        if (t >= 0)
            return 1.0 / (1.0 + Math.Exp(-t));
        var e = Math.Exp(t);
        return e / (1.0 + e);
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
