namespace IpoptNet.Modelling;

public sealed class Log : Expr
{
    public Expr Argument { get; set; }
    private double[]? _gradBuffer;

    public Log(Expr argument) => Argument = argument;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Log(Argument.Evaluate(x));

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateGradientCompact(x, compactGrad, multiplier / arg, sortedVarIndices);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, hess, multiplier / arg);

        var secondDeriv = -1.0 / (arg * arg);
        var coeff = multiplier * secondDeriv;
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

    protected override Expr CloneCore() => new Log(Argument);

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
