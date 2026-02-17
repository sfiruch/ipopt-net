using System.Text;

namespace IpoptNet.Modelling;

internal sealed class NegationNode : ExprNode
{
    public ExprNode Operand { get; set; }

    public NegationNode(ExprNode operand)
    {
        Operand = operand;
    }

    internal override double Evaluate(ReadOnlySpan<double> x)
    {
        var val = Operand.Evaluate(x);
        return -val;
    }

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        Operand.AccumulateGradientCompact(x, compactGrad, -multiplier, sortedVarIndices);
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        Operand.AccumulateHessian(x, hess, -multiplier);
    }

    internal override void CollectVariables(HashSet<Variable> variables) => Operand.CollectVariables(variables);
    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries) => Operand.CollectHessianSparsity(entries);
    internal override bool IsConstantWrtX() => Operand.IsConstantWrtX();
    internal override bool IsLinear() => Operand.IsLinear();
    internal override bool IsAtMostQuadratic() => Operand.IsAtMostQuadratic();

    internal override void PrepareChildren()
    {
        Operand.Prepare();
    }

    internal override void ClearChildren()
    {
        Operand.Clear();
    }

    public override string ToString()
    {
        // If operand is simple, format inline
        if (Operand.IsSimpleForPrinting())
            return $"Negation: -({Operand})";

        // Otherwise, use multi-line tree format
        var sb = new StringBuilder();
        sb.AppendLine("Negation:");
        foreach (var line in Operand.ToString()!.Split(Environment.NewLine))
            sb.AppendLine($"  {line}");
        return sb.ToString().TrimEnd();
    }
}
