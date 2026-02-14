using System.Text;

namespace IpoptNet.Modelling;

public sealed class Negation : Expr
{
    public Expr Operand { get; set; }

    public Negation(Expr operand)
    {
        Operand = operand;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var val = Operand.Evaluate(x);
        return -val;
    }

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, Dictionary<int, int> varIndexToCompact)
    {
        Operand.AccumulateGradientCompact(x, compactGrad, -multiplier, varIndexToCompact);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        Operand.AccumulateHessian(x, hess, -multiplier);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Operand.CollectVariables(variables);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries) => Operand.CollectHessianSparsity(entries);
    protected override bool IsConstantWrtXCore() => Operand.IsConstantWrtX();
    protected override bool IsLinearCore() => Operand.IsLinear();
    protected override bool IsAtMostQuadraticCore() => Operand.IsAtMostQuadratic();

    protected override Expr CloneCore() => new Negation(Operand);

    protected override void PrepareChildren()
    {
        Operand.Prepare();
    }

    protected override void ClearChildren()
    {
        Operand.Clear();
    }

    protected override string ToStringCore()
    {
        // If operand is simple, format inline
        if (Operand.IsSimpleForPrinting())
            return $"Negation: -({Operand})";

        // Otherwise, use multi-line tree format
        var sb = new StringBuilder();
        sb.AppendLine("Negation:");
        foreach (var line in Operand.ToString().Split(Environment.NewLine))
            sb.AppendLine($"  {line}");
        return sb.ToString().TrimEnd();
    }
}
