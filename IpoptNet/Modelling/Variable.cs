namespace IpoptNet.Modelling;

public sealed class Variable : Expr
{
    public int Index { get; internal set; } = -1;
    public double LowerBound { get; set; } = double.NegativeInfinity;
    public double UpperBound { get; set; } = double.PositiveInfinity;

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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        // Variable has no second derivative contribution
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => variables.Add(this);

    protected override Expr CloneCore() => this; // Variables are singletons - return self
}
