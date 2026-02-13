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

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        Operand.AccumulateGradient(x, grad, -multiplier);
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

    protected override void CacheVariablesForChildren()
    {
        Operand.CacheVariables();
    }

    protected override void ClearCachedVariablesForChildren()
    {
        Operand.ClearCachedVariables();
    }

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}Negation:");
        Operand.Print(writer, indent + "  ");
    }
}
