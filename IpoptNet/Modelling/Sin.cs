using System.Buffers;

namespace IpoptNet.Modelling;

public sealed class Sin : Expr
{
    public Expr Argument { get; set; }

    public Sin(Expr argument) => Argument = argument;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Sin(Argument.Evaluate(x));

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateGradient(x, grad, multiplier * Math.Cos(arg));
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, hess, multiplier * Math.Cos(arg));

        var n = x.Length;
        var gradArg = ArrayPool<double>.Shared.Rent(n);

        var vars = Argument._cachedVariables!;

        if (vars.Count < n / 32)
            foreach (var v in vars) gradArg[v.Index] = 0;
        else
            Array.Clear(gradArg);

        Argument.AccumulateGradient(x, gradArg, 1.0);
        AccumulateOuterHessian(x, hess, multiplier * -Math.Sin(arg), vars, gradArg);

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

    protected override Expr CloneCore() => new Sin(Argument);

    protected override void CacheVariablesForChildren()
    {
        Argument.CacheVariables();
    }

    protected override void ClearCachedVariablesForChildren()
    {
        Argument.ClearCachedVariables();
    }
}
