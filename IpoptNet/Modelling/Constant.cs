namespace IpoptNet.Modelling;

public sealed class Constant : Expr
{
    public double Value { get; set; }

    public Constant(double value) => Value = value;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Value;
    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, Dictionary<int, int> varIndexToCompact) { }
    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier) { }
    protected override void CollectVariablesCore(HashSet<Variable> variables) { }
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries) { }
    protected override bool IsConstantWrtXCore() => true;
    protected override bool IsLinearCore() => true;
    protected override bool IsAtMostQuadraticCore() => true;

    protected override Expr CloneCore() => new Constant(Value);

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}Constant: {Value}");
    }
}
