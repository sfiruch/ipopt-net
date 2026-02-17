namespace IpoptNet.Modelling;

internal sealed class ConstantNode : ExprNode
{
    public double Value { get; set; }

    public ConstantNode(double value) => Value = value;

    internal override double Evaluate(ReadOnlySpan<double> x) => Value;
    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices) { }
    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier) { }
    internal override void CollectVariables(HashSet<Variable> variables) { }
    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries) { }
    internal override bool IsConstantWrtX() => true;
    internal override bool IsLinear() => true;
    internal override bool IsAtMostQuadratic() => true;

    public override string ToString() => Value.ToString();

    internal override bool IsSimpleForPrinting() => true;
}
