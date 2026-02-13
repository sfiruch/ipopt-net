using System.Buffers;

namespace IpoptNet.Modelling;

public sealed class Tan : Expr
{
    public Expr Argument { get; set; }

    public Tan(Expr argument) => Argument = argument;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Tan(Argument.Evaluate(x));

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        var cos = Math.Cos(arg);
        Argument.AccumulateGradient(x, grad, multiplier / (cos * cos));
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        var cos = Math.Cos(arg);
        Argument.AccumulateHessian(x, hess, multiplier / (cos * cos));

        var n = x.Length;
        var gradArg = ArrayPool<double>.Shared.Rent(n);

        var vars = Argument._cachedVariables!;

        if (vars.Count < n / 32)
            foreach (var v in vars) gradArg[v.Index] = 0;
        else
            Array.Clear(gradArg);

        Argument.AccumulateGradient(x, gradArg, 1.0);
        var secondDeriv = 2 * Math.Tan(arg) / (cos * cos);
        AccumulateOuterHessian(x, hess, multiplier * secondDeriv, vars, gradArg);

        ArrayPool<double>.Shared.Return(gradArg);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Argument.CollectVariables(variables);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        if (!Argument.IsConstantWrtX())
        {
            AddClique(entries, Argument._cachedVariables!);
        }
    }
    protected override bool IsConstantWrtXCore() => Argument.IsConstantWrtX();
    protected override bool IsLinearCore() => Argument.IsConstantWrtX();
    protected override bool IsAtMostQuadraticCore() => Argument.IsConstantWrtX();

    protected override Expr CloneCore() => new Tan(Argument);

    protected override void CacheVariablesForChildren()
    {
        Argument.CacheVariables();
    }

    protected override void ClearCachedVariablesForChildren()
    {
        Argument.ClearCachedVariables();
    }
}
