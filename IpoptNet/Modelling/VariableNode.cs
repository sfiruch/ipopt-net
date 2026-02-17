namespace IpoptNet.Modelling;

internal sealed class VariableNode : ExprNode
{
    public Variable Variable { get; }

    public VariableNode(Variable variable)
    {
        Variable = variable;
    }

    internal override double Evaluate(ReadOnlySpan<double> x) => x[Variable.Index];

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        compactGrad[Array.BinarySearch(sortedVarIndices, Variable.Index)] += multiplier;
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // Variable has no second derivative contribution
    }

    internal override void CollectVariables(HashSet<Variable> variables) => variables.Add(Variable);
    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries) { }
    internal override bool IsConstantWrtX() => false;
    internal override bool IsLinear() => true;
    internal override bool IsAtMostQuadratic() => true;

    public override string ToString() => $"x[{Variable.Index}]";

    internal override bool IsSimpleForPrinting() => true;
}
