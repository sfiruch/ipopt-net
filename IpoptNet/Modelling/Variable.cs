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

    public override double Evaluate(ReadOnlySpan<double> x) => x[Index];

    public override void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        grad[Index] += multiplier;
    }

    public override void AccumulateHessian(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        // Variable has no second derivative contribution
    }

    public override void CollectVariables(HashSet<Variable> variables) => variables.Add(this);
}
