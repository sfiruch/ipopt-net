namespace IpoptNet.Modelling;

public sealed class Variable : Expr
{
    public int Index { get; internal set; } = -1;
    public double LowerBound { get; set; } = double.NegativeInfinity;
    public double UpperBound { get; set; } = double.PositiveInfinity;
    public double Start { get; set; } = 0;
    public double LowerBoundDualStart { get; set; } = 0;
    public double UpperBoundDualStart { get; set; } = 0;

    internal Variable() { }

    public Variable(double lowerBound, double upperBound)
    {
        LowerBound = lowerBound;
        UpperBound = upperBound;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x) => x[Index];

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        grad[Index] += multiplier;
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // Variable has no second derivative contribution
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => variables.Add(this);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries) { }
    protected override bool IsConstantWrtXCore() => false;
    protected override bool IsLinearCore() => true;
    protected override bool IsAtMostQuadraticCore() => true;

    protected override Expr CloneCore() => this; // Variables are singletons - return self
}
