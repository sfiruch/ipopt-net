namespace IpoptNet.Modelling;

public sealed class Variable : Expr
{
    public int Index { get; internal set; } = -1;
    public double LowerBound = double.NegativeInfinity;
    public double UpperBound = double.PositiveInfinity;
    public double? Start;
    public double LowerBoundDualStart = 0;
    public double UpperBoundDualStart = 0;

    internal Variable() { }

    public Variable(double lowerBound, double upperBound)
    {
        LowerBound = lowerBound;
        UpperBound = upperBound;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x) => x[Index];

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        compactGrad[Array.BinarySearch(sortedVarIndices, Index)] += multiplier;
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

    protected override string ToStringCore() => $"x[{Index}]";

    internal override bool IsSimpleForPrinting() => true;
}
