using System.Runtime.CompilerServices;

namespace IpoptNet.Modelling;

public abstract class Expr
{
    protected Expr? _replacement;

    public double Evaluate(ReadOnlySpan<double> x) => _replacement?.Evaluate(x) ?? EvaluateCore(x);
    public void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad, double multiplier) =>
        (_replacement ?? this).AccumulateGradientCore(x, grad, multiplier);
    public void AccumulateHessian(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier) =>
        (_replacement ?? this).AccumulateHessianCore(x, grad, hess, multiplier);
    public void CollectVariables(HashSet<Variable> variables) =>
        (_replacement ?? this).CollectVariablesCore(variables);

    protected abstract double EvaluateCore(ReadOnlySpan<double> x);
    protected abstract void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier);
    protected abstract void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier);
    protected abstract void CollectVariablesCore(HashSet<Variable> variables);

    public override bool Equals(object? obj) => ReferenceEquals(this, obj);
    public override int GetHashCode() => RuntimeHelpers.GetHashCode(this);

    public static implicit operator Expr(int value) => new Constant(value);
    public static implicit operator Expr(double value) => new Constant(value);

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

    // C# 14 compound assignment operators - modify expression in-place for efficiency
    public void operator +=(Expr other)
    {
        // Check if we've been replaced with a Sum
        if (_replacement is Sum sum)
            sum.Terms.Add(other);
        // Check if this is a zero constant with no replacement yet
        else if (_replacement == null && this is Constant { Value: 0 })
            ReplaceWith(other);
        // Otherwise create a new Sum with current value and new term
        else
            ReplaceWith(new Sum([Clone(), other]));
    }

    public void operator -=(Expr other)
    {
        // Check if we've been replaced with a Sum
        if (_replacement is Sum sum)
            sum.Terms.Add(-other);
        // Check if this is a zero constant with no replacement yet
        else if (_replacement == null && this is Constant { Value: 0 })
            ReplaceWith(-other);
        // Otherwise create a new Sum with current value and negated term
        else
            ReplaceWith(new Sum([Clone(), -other]));
    }

    public void operator *=(Expr other)
    {
        // Check if we've been replaced with a Product
        if (_replacement is Product product)
            product.Factors.Add(other);
        // Check if this is a one constant with no replacement yet
        else if (_replacement == null && this is Constant { Value: 1 })
            ReplaceWith(other);
        // Otherwise create a new Product with current value and new factor
        else
            ReplaceWith(new Product([Clone(), other]));
    }

    public void operator /=(Expr other)
    {
        // Check if we've been replaced with a Product (add reciprocal)
        if (_replacement is Product product)
            product.Factors.Add(new BinaryOp(1, other, BinaryOpKind.Divide));
        // Otherwise create a division
        else
            ReplaceWith(new BinaryOp(Clone(), other, BinaryOpKind.Divide));
    }

    protected Expr Clone()
    {
        if (_replacement != null)
            return _replacement.Clone();
        return CloneCore();
    }

    protected abstract Expr CloneCore();

    protected void ReplaceWith(Expr other)
    {
        _replacement = other;
    }

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
}

public sealed class Constant : Expr
{
    public double Value { get; set; }

    public Constant(double value) => Value = value;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Value;
    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier) { }
    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier) { }
    protected override void CollectVariablesCore(HashSet<Variable> variables) { }

    protected override Expr CloneCore() => new Constant(Value);
}

public enum BinaryOpKind { Add, Subtract, Multiply, Divide }

public sealed class BinaryOp : Expr
{
    public Expr Left { get; set; }
    public Expr Right { get; set; }
    public BinaryOpKind Kind { get; set; }

    public BinaryOp(Expr left, Expr right, BinaryOpKind kind)
    {
        Left = left;
        Right = right;
        Kind = kind;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
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

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
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

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        Left.CollectVariables(variables);
        Right.CollectVariables(variables);
    }

    protected override Expr CloneCore() => new BinaryOp(Left, Right, Kind);
}

public enum UnaryOpKind { Negate }

public sealed class UnaryOp : Expr
{
    public Expr Operand { get; set; }
    public UnaryOpKind Kind { get; set; }

    public UnaryOp(Expr operand, UnaryOpKind kind)
    {
        Operand = operand;
        Kind = kind;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var val = Operand.Evaluate(x);
        return Kind switch
        {
            UnaryOpKind.Negate => -val,
            _ => throw new InvalidOperationException()
        };
    }

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        switch (Kind)
        {
            case UnaryOpKind.Negate:
                Operand.AccumulateGradient(x, grad, -multiplier);
                break;
        }
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        switch (Kind)
        {
            case UnaryOpKind.Negate:
                Operand.AccumulateHessian(x, grad, hess, -multiplier);
                break;
        }
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Operand.CollectVariables(variables);

    protected override Expr CloneCore() => new UnaryOp(Operand, Kind);
}

public sealed class PowerOp : Expr
{
    public Expr Base { get; set; }
    public double Exponent { get; set; }

    public PowerOp(Expr @base, double exponent)
    {
        Base = @base;
        Exponent = exponent;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Pow(Base.Evaluate(x), Exponent);

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        // d(b^n)/dx = n * b^(n-1) * db/dx
        var bVal = Base.Evaluate(x);
        var deriv = Exponent * Math.Pow(bVal, Exponent - 1);
        Base.AccumulateGradient(x, grad, multiplier * deriv);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
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

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Base.CollectVariables(variables);

    protected override Expr CloneCore() => new PowerOp(Base, Exponent);
}

public enum MathFunction { Sin, Cos, Tan, Exp, Log }

public sealed class FunctionCall : Expr
{
    public Expr Argument { get; set; }
    public MathFunction Function { get; set; }

    public FunctionCall(Expr argument, MathFunction function)
    {
        Argument = argument;
        Function = function;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
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

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
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

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Argument.CollectVariables(variables);

    protected override Expr CloneCore() => new FunctionCall(Argument, Function);
}

public sealed class Sum : Expr
{
    public List<Expr> Terms { get; set; }

    public Sum() => Terms = [];
    public Sum(List<Expr> terms) => Terms = terms;

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var result = 0.0;
        foreach (var term in Terms)
            result += term.Evaluate(x);
        return result;
    }

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        foreach (var term in Terms)
            term.AccumulateGradient(x, grad, multiplier);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        foreach (var term in Terms)
            term.AccumulateHessian(x, grad, hess, multiplier);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        foreach (var term in Terms)
            term.CollectVariables(variables);
    }

    protected override Expr CloneCore() => new Sum([.. Terms]);
}

public sealed class Product : Expr
{
    public List<Expr> Factors { get; set; }

    public Product() => Factors = [];
    public Product(List<Expr> factors) => Factors = factors;

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var result = 1.0;
        foreach (var factor in Factors)
            result *= factor.Evaluate(x);
        return result;
    }

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        // Product rule: d(f*g*h)/dx = df/dx*g*h + f*dg/dx*h + f*g*dh/dx
        foreach (var factor in Factors)
        {
            var otherProduct = 1.0;
            foreach (var other in Factors)
            {
                if (other != factor)
                    otherProduct *= other.Evaluate(x);
            }
            factor.AccumulateGradient(x, grad, multiplier * otherProduct);
        }
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        // Simplified implementation: convert to nested BinaryOps for Hessian calculation
        if (Factors.Count == 0)
            return;
        if (Factors.Count == 1)
        {
            Factors[0].AccumulateHessian(x, grad, hess, multiplier);
            return;
        }

        var result = Factors[0];
        for (int i = 1; i < Factors.Count; i++)
            result = new BinaryOp(result, Factors[i], BinaryOpKind.Multiply);
        result.AccumulateHessian(x, grad, hess, multiplier);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        foreach (var factor in Factors)
            factor.CollectVariables(variables);
    }

    protected override Expr CloneCore() => new Product([.. Factors]);
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
