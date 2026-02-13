using System.Buffers;

namespace IpoptNet.Modelling;

public sealed class Log : Expr
{
    public Expr Argument { get; set; }

    public Log(Expr argument) => Argument = argument;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Log(Argument.Evaluate(x));

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateGradient(x, grad, multiplier / arg);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, hess, multiplier / arg);

        var n = x.Length;
        var gradArg = ArrayPool<double>.Shared.Rent(n);

        var vars = Argument._cachedVariables!;

        if (vars.Count < n / 32)
            foreach (var v in vars) gradArg[v.Index] = 0;
        else
            Array.Clear(gradArg);

        Argument.AccumulateGradient(x, gradArg, 1.0);
        var secondDeriv = -1.0 / (arg * arg);
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

    protected override Expr CloneCore() => new Log(Argument);

    protected override void CacheVariablesForChildren()
    {
        Argument.CacheVariables();
    }

    protected override void ClearCachedVariablesForChildren()
    {
        Argument.ClearCachedVariables();
    }
}
