using System.Runtime.CompilerServices;

namespace IpoptNet.Modelling;

public abstract class Expr
{
    public abstract double Evaluate(ReadOnlySpan<double> x);
    public abstract void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad, double multiplier);
    public abstract void AccumulateHessian(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier);
    public abstract void CollectVariables(HashSet<Variable> variables);

    public override bool Equals(object? obj) => ReferenceEquals(this, obj);
    public override int GetHashCode() => RuntimeHelpers.GetHashCode(this);

    public static Expr operator +(Expr a, Expr b) => new BinaryOp(a, b, BinaryOpKind.Add);
    public static Expr operator -(Expr a, Expr b) => new BinaryOp(a, b, BinaryOpKind.Subtract);
    public static Expr operator *(Expr a, Expr b) => new BinaryOp(a, b, BinaryOpKind.Multiply);
    public static Expr operator /(Expr a, Expr b) => new BinaryOp(a, b, BinaryOpKind.Divide);
    public static Expr operator -(Expr a) => new UnaryOp(a, UnaryOpKind.Negate);

    public static Expr operator +(Expr a, double b) => new BinaryOp(a, new Constant(b), BinaryOpKind.Add);
    public static Expr operator +(double a, Expr b) => new BinaryOp(new Constant(a), b, BinaryOpKind.Add);
    public static Expr operator -(Expr a, double b) => new BinaryOp(a, new Constant(b), BinaryOpKind.Subtract);
    public static Expr operator -(double a, Expr b) => new BinaryOp(new Constant(a), b, BinaryOpKind.Subtract);
    public static Expr operator *(Expr a, double b) => new BinaryOp(a, new Constant(b), BinaryOpKind.Multiply);
    public static Expr operator *(double a, Expr b) => new BinaryOp(new Constant(a), b, BinaryOpKind.Multiply);
    public static Expr operator /(Expr a, double b) => new BinaryOp(a, new Constant(b), BinaryOpKind.Divide);
    public static Expr operator /(double a, Expr b) => new BinaryOp(new Constant(a), b, BinaryOpKind.Divide);

    public static Constraint operator >=(Expr expr, double value) => new(expr, value, double.PositiveInfinity);
    public static Constraint operator <=(Expr expr, double value) => new(expr, double.NegativeInfinity, value);
    public static Constraint operator ==(Expr expr, double value) => new(expr, value, value);
    public static Constraint operator !=(Expr expr, double value) => throw new NotSupportedException("Inequality constraints (!=) are not supported in optimization models.");
    public static Constraint operator >(Expr expr, double value) => new(expr, value, double.PositiveInfinity);
    public static Constraint operator <(Expr expr, double value) => new(expr, double.NegativeInfinity, value);

    public static Constraint operator >=(double value, Expr expr) => new(expr, double.NegativeInfinity, value);
    public static Constraint operator <=(double value, Expr expr) => new(expr, value, double.PositiveInfinity);
    public static Constraint operator ==(double value, Expr expr) => new(expr, value, value);
    public static Constraint operator !=(double value, Expr expr) => throw new NotSupportedException("Inequality constraints (!=) are not supported in optimization models.");
    public static Constraint operator >(double value, Expr expr) => new(expr, double.NegativeInfinity, value);
    public static Constraint operator <(double value, Expr expr) => new(expr, value, double.PositiveInfinity);

    public static Expr Pow(Expr @base, double exponent) => new PowerOp(@base, exponent);
    public static Expr Pow(Expr @base, Expr exponent) => Exp(exponent * Log(@base));
    public static Expr Sqrt(Expr a) => new PowerOp(a, 0.5);
    public static Expr Sin(Expr a) => new FunctionCall(a, MathFunction.Sin);
    public static Expr Cos(Expr a) => new FunctionCall(a, MathFunction.Cos);
    public static Expr Tan(Expr a) => new FunctionCall(a, MathFunction.Tan);
    public static Expr Exp(Expr a) => new FunctionCall(a, MathFunction.Exp);
    public static Expr Log(Expr a) => new FunctionCall(a, MathFunction.Log);

    public static implicit operator Expr(double value) => new Constant(value);
}

public sealed class Constant : Expr
{
    public double Value { get; }

    public Constant(double value) => Value = value;

    public override double Evaluate(ReadOnlySpan<double> x) => Value;
    public override void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad, double multiplier) { }
    public override void AccumulateHessian(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier) { }
    public override void CollectVariables(HashSet<Variable> variables) { }
}

public enum BinaryOpKind { Add, Subtract, Multiply, Divide }

public sealed class BinaryOp : Expr
{
    public Expr Left { get; }
    public Expr Right { get; }
    public BinaryOpKind Kind { get; }

    public BinaryOp(Expr left, Expr right, BinaryOpKind kind)
    {
        Left = left;
        Right = right;
        Kind = kind;
    }

    public override double Evaluate(ReadOnlySpan<double> x)
    {
        var l = Left.Evaluate(x);
        var r = Right.Evaluate(x);
        return Kind switch
        {
            BinaryOpKind.Add => l + r,
            BinaryOpKind.Subtract => l - r,
            BinaryOpKind.Multiply => l * r,
            BinaryOpKind.Divide => l / r,
            _ => throw new InvalidOperationException()
        };
    }

    public override void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        switch (Kind)
        {
            case BinaryOpKind.Add:
                Left.AccumulateGradient(x, grad, multiplier);
                Right.AccumulateGradient(x, grad, multiplier);
                break;
            case BinaryOpKind.Subtract:
                Left.AccumulateGradient(x, grad, multiplier);
                Right.AccumulateGradient(x, grad, -multiplier);
                break;
            case BinaryOpKind.Multiply:
                var rVal = Right.Evaluate(x);
                var lVal = Left.Evaluate(x);
                Left.AccumulateGradient(x, grad, multiplier * rVal);
                Right.AccumulateGradient(x, grad, multiplier * lVal);
                break;
            case BinaryOpKind.Divide:
                var rValDiv = Right.Evaluate(x);
                var lValDiv = Left.Evaluate(x);
                Left.AccumulateGradient(x, grad, multiplier / rValDiv);
                Right.AccumulateGradient(x, grad, -multiplier * lValDiv / (rValDiv * rValDiv));
                break;
        }
    }

    public override void AccumulateHessian(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        switch (Kind)
        {
            case BinaryOpKind.Add:
                Left.AccumulateHessian(x, grad, hess, multiplier);
                Right.AccumulateHessian(x, grad, hess, multiplier);
                break;
            case BinaryOpKind.Subtract:
                Left.AccumulateHessian(x, grad, hess, multiplier);
                Right.AccumulateHessian(x, grad, hess, -multiplier);
                break;
            case BinaryOpKind.Multiply:
                // d²(L*R)/dx² = d²L/dx² * R + 2 * dL/dx * dR/dx + L * d²R/dx²
                var rVal = Right.Evaluate(x);
                var lVal = Left.Evaluate(x);
                Left.AccumulateHessian(x, grad, hess, multiplier * rVal);
                Right.AccumulateHessian(x, grad, hess, multiplier * lVal);
                // Cross term: dL/dx * dR/dx
                var n = grad.Length;
                Span<double> gradL = stackalloc double[n];
                Span<double> gradR = stackalloc double[n];
                Left.AccumulateGradient(x, gradL, 1.0);
                Right.AccumulateGradient(x, gradR, 1.0);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j <= i; j++)
                        hess.Add(i, j, multiplier * (gradL[i] * gradR[j] + gradL[j] * gradR[i]));
                break;
            case BinaryOpKind.Divide:
                // d(L/R)/dx = (dL/dx * R - L * dR/dx) / R²
                // d²(L/R)/dx² = d²L/dx²/R - L*d²R/dx²/R² - 2*dL/dx*dR/dx/R² + 2*L*(dR/dx)²/R³
                var rValDiv = Right.Evaluate(x);
                var lValDiv = Left.Evaluate(x);
                var r2 = rValDiv * rValDiv;
                var r3 = r2 * rValDiv;
                Left.AccumulateHessian(x, grad, hess, multiplier / rValDiv);
                Right.AccumulateHessian(x, grad, hess, -multiplier * lValDiv / r2);
                var nDiv = grad.Length;
                Span<double> gradLDiv = stackalloc double[nDiv];
                Span<double> gradRDiv = stackalloc double[nDiv];
                Left.AccumulateGradient(x, gradLDiv, 1.0);
                Right.AccumulateGradient(x, gradRDiv, 1.0);
                for (int i = 0; i < nDiv; i++)
                    for (int j = 0; j <= i; j++)
                    {
                        var cross = -multiplier / r2 * (gradLDiv[i] * gradRDiv[j] + gradLDiv[j] * gradRDiv[i]);
                        var rr = multiplier * 2 * lValDiv / r3 * gradRDiv[i] * gradRDiv[j];
                        hess.Add(i, j, cross + rr);
                    }
                break;
        }
    }

    public override void CollectVariables(HashSet<Variable> variables)
    {
        Left.CollectVariables(variables);
        Right.CollectVariables(variables);
    }
}

public enum UnaryOpKind { Negate }

public sealed class UnaryOp : Expr
{
    public Expr Operand { get; }
    public UnaryOpKind Kind { get; }

    public UnaryOp(Expr operand, UnaryOpKind kind)
    {
        Operand = operand;
        Kind = kind;
    }

    public override double Evaluate(ReadOnlySpan<double> x)
    {
        var val = Operand.Evaluate(x);
        return Kind switch
        {
            UnaryOpKind.Negate => -val,
            _ => throw new InvalidOperationException()
        };
    }

    public override void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        switch (Kind)
        {
            case UnaryOpKind.Negate:
                Operand.AccumulateGradient(x, grad, -multiplier);
                break;
        }
    }

    public override void AccumulateHessian(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        switch (Kind)
        {
            case UnaryOpKind.Negate:
                Operand.AccumulateHessian(x, grad, hess, -multiplier);
                break;
        }
    }

    public override void CollectVariables(HashSet<Variable> variables) => Operand.CollectVariables(variables);
}

public sealed class PowerOp : Expr
{
    public Expr Base { get; }
    public double Exponent { get; }

    public PowerOp(Expr @base, double exponent)
    {
        Base = @base;
        Exponent = exponent;
    }

    public override double Evaluate(ReadOnlySpan<double> x) => Math.Pow(Base.Evaluate(x), Exponent);

    public override void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        // d(b^n)/dx = n * b^(n-1) * db/dx
        var bVal = Base.Evaluate(x);
        var deriv = Exponent * Math.Pow(bVal, Exponent - 1);
        Base.AccumulateGradient(x, grad, multiplier * deriv);
    }

    public override void AccumulateHessian(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        // d²(b^n)/dx² = n*(n-1)*b^(n-2)*(db/dx)² + n*b^(n-1)*d²b/dx²
        var bVal = Base.Evaluate(x);
        var firstDerivCoeff = Exponent * Math.Pow(bVal, Exponent - 1);
        var secondDerivCoeff = Exponent * (Exponent - 1) * Math.Pow(bVal, Exponent - 2);
        Base.AccumulateHessian(x, grad, hess, multiplier * firstDerivCoeff);
        var n = grad.Length;
        Span<double> gradB = stackalloc double[n];
        Base.AccumulateGradient(x, gradB, 1.0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                hess.Add(i, j, multiplier * secondDerivCoeff * gradB[i] * gradB[j]);
    }

    public override void CollectVariables(HashSet<Variable> variables) => Base.CollectVariables(variables);
}

public enum MathFunction { Sin, Cos, Tan, Exp, Log }

public sealed class FunctionCall : Expr
{
    public Expr Argument { get; }
    public MathFunction Function { get; }

    public FunctionCall(Expr argument, MathFunction function)
    {
        Argument = argument;
        Function = function;
    }

    public override double Evaluate(ReadOnlySpan<double> x)
    {
        var arg = Argument.Evaluate(x);
        return Function switch
        {
            MathFunction.Sin => Math.Sin(arg),
            MathFunction.Cos => Math.Cos(arg),
            MathFunction.Tan => Math.Tan(arg),
            MathFunction.Exp => Math.Exp(arg),
            MathFunction.Log => Math.Log(arg),
            _ => throw new InvalidOperationException()
        };
    }

    public override void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        var deriv = Function switch
        {
            MathFunction.Sin => Math.Cos(arg),
            MathFunction.Cos => -Math.Sin(arg),
            MathFunction.Tan => 1.0 / (Math.Cos(arg) * Math.Cos(arg)),
            MathFunction.Exp => Math.Exp(arg),
            MathFunction.Log => 1.0 / arg,
            _ => throw new InvalidOperationException()
        };
        Argument.AccumulateGradient(x, grad, multiplier * deriv);
    }

    public override void AccumulateHessian(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        var (firstDeriv, secondDeriv) = Function switch
        {
            MathFunction.Sin => (Math.Cos(arg), -Math.Sin(arg)),
            MathFunction.Cos => (-Math.Sin(arg), -Math.Cos(arg)),
            MathFunction.Tan => (1.0 / (Math.Cos(arg) * Math.Cos(arg)), 2 * Math.Tan(arg) / (Math.Cos(arg) * Math.Cos(arg))),
            MathFunction.Exp => (Math.Exp(arg), Math.Exp(arg)),
            MathFunction.Log => (1.0 / arg, -1.0 / (arg * arg)),
            _ => throw new InvalidOperationException()
        };
        Argument.AccumulateHessian(x, grad, hess, multiplier * firstDeriv);
        var n = grad.Length;
        Span<double> gradArg = stackalloc double[n];
        Argument.AccumulateGradient(x, gradArg, 1.0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                hess.Add(i, j, multiplier * secondDeriv * gradArg[i] * gradArg[j]);
    }

    public override void CollectVariables(HashSet<Variable> variables) => Argument.CollectVariables(variables);
}

public sealed class HessianAccumulator
{
    private readonly Dictionary<(int, int), double> _entries = new();
    private readonly int _n;

    public HessianAccumulator(int n) => _n = n;

    public void Add(int i, int j, double value)
    {
        if (Math.Abs(value) < 1e-15) return;
        var key = i >= j ? (i, j) : (j, i); // Lower triangular
        _entries.TryGetValue(key, out var existing);
        _entries[key] = existing + value;
    }

    public IReadOnlyDictionary<(int row, int col), double> Entries => _entries;

    public void Clear() => _entries.Clear();

    public int GetNonZeroCount()
    {
        // For IPOPT, we need the structure of the lower triangular Hessian
        var count = 0;
        foreach (var entry in _entries)
            if (Math.Abs(entry.Value) >= 1e-15)
                count++;
        return count;
    }
}
