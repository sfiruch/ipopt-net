namespace IpoptNet.Modelling;

public sealed class Sin : Expr
{
    public Expr Argument { get; set; }
    private double[]? _gradBuffer;

    public Sin(Expr argument) => Argument = argument;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Sin(Argument.Evaluate(x));

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateGradientCompact(x, compactGrad, multiplier * Math.Cos(arg), sortedVarIndices);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, hess, multiplier * Math.Cos(arg));

        var coeff = multiplier * -Math.Sin(arg);
        if (Math.Abs(coeff) < 1e-18) return;

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

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Argument.CollectVariables(variables);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        if (!Argument.IsConstantWrtX())
        {
            AddClique(entries, Argument._cachedVariables!);
        }
    }
    protected override bool IsConstantWrtXCore() => Argument.IsConstantWrtX();
    protected override bool IsLinearCore() => Argument.IsConstantWrtX();
    protected override bool IsAtMostQuadraticCore() => Argument.IsConstantWrtX();

    protected override Expr CloneCore() => new Sin(Argument);

    protected override void PrepareChildren()
    {
        Argument.Prepare();
        _gradBuffer = new double[Argument._cachedVariables!.Count];
    }

    protected override void ClearChildren()
    {
        Argument.Clear();
        _gradBuffer = null;
    }
}
