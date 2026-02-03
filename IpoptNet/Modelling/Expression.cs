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
    public void CollectHessianSparsity(HashSet<(int row, int col)> entries) =>
        (_replacement ?? this).CollectHessianSparsityCore(entries);

    /// <summary>
    /// Returns true if this expression contains no variables (is a constant value).
    /// </summary>
    public bool IsConstantWrtX() => (_replacement ?? this).IsConstantWrtXCore();

    /// <summary>
    /// Returns true if this expression is linear (constant gradient, zero Hessian).
    /// Examples: 2*x + 3*y - 5, x, Constant
    /// </summary>
    public bool IsLinear() => (_replacement ?? this).IsLinearCore();

    /// <summary>
    /// Returns true if this expression is at most quadratic (constant Hessian).
    /// Examples: x*y, x^2, 2*x + 3*y - 5
    /// </summary>
    public bool IsAtMostQuadratic() => (_replacement ?? this).IsAtMostQuadraticCore();

    protected abstract double EvaluateCore(ReadOnlySpan<double> x);
    protected abstract void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier);
    protected abstract void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier);
    protected abstract void CollectVariablesCore(HashSet<Variable> variables);
    protected abstract void CollectHessianSparsityCore(HashSet<(int row, int col)> entries);
    protected abstract bool IsConstantWrtXCore();
    protected abstract bool IsLinearCore();
    protected abstract bool IsAtMostQuadraticCore();

    protected static void AddSparsityEntry(HashSet<(int row, int col)> entries, int i, int j)
    {
        if (i < j) (i, j) = (j, i);
        entries.Add((i, j));
    }

    protected static void AddClique(HashSet<(int row, int col)> entries, HashSet<Variable> variables)
    {
        var vars = variables.ToArray();
        for (int i = 0; i < vars.Length; i++)
            for (int j = 0; j <= i; j++)
                AddSparsityEntry(entries, vars[i].Index, vars[j].Index);
    }

    public override bool Equals(object? obj) => ReferenceEquals(this, obj);
    public override int GetHashCode() => RuntimeHelpers.GetHashCode(this);

    public static implicit operator Expr(int value) => new Constant(value);
    public static implicit operator Expr(double value) => new Constant(value);

    public static Expr operator +(Expr a, Expr b)
    {
        // If left operand is already a Sum, extend it with the right operand
        if (a is Sum sumA)
        {
            var newTerms = new List<Expr>(sumA.Terms.Count + 1);
            newTerms.AddRange(sumA.Terms);
            newTerms.Add(b);
            return new Sum(newTerms);
        }
        // If right operand is a Sum, prepend the left operand to it
        else if (b is Sum sumB)
        {
            var newTerms = new List<Expr>(sumB.Terms.Count + 1);
            newTerms.Add(a);
            newTerms.AddRange(sumB.Terms);
            return new Sum(newTerms);
        }
        // Always use Sum for addition to avoid deep Division trees (Sum is semantically equivalent)
        else
        {
            return new Sum([a, b]);
        }
    }
    public static Expr operator -(Expr a, Expr b)
    {
        // If left operand is already a Sum, extend it with the negated right operand
        if (a is Sum sumA)
        {
            var newTerms = new List<Expr>(sumA.Terms.Count + 1);
            newTerms.AddRange(sumA.Terms);
            newTerms.Add(-b);
            return new Sum(newTerms);
        }
        // Create a new Sum with both terms to avoid building deep Division trees
        else
        {
            return new Sum([a, -b]);
        }
    }
    public static Expr operator *(Expr a, Expr b)
    {
        // If left operand is already a Product, extend it with the right operand
        if (a is Product prodA)
        {
            var newFactors = new List<Expr>(prodA.Factors.Count + 1);
            newFactors.AddRange(prodA.Factors);
            newFactors.Add(b);
            return new Product(newFactors);
        }
        // If right operand is a Product, prepend the left operand to it
        else if (b is Product prodB)
        {
            var newFactors = new List<Expr>(prodB.Factors.Count + 1);
            newFactors.Add(a);
            newFactors.AddRange(prodB.Factors);
            return new Product(newFactors);
        }
        // Always use Product for multiplication to avoid building deep Division trees
        else
        {
            return new Product([a, b]);
        }
    }
    public static Expr operator /(Expr a, Expr b) => new Division(a, b);
    public static Expr operator -(Expr a) => new Negation(a);

    public static Expr operator +(Expr a, double b)
    {
        if (a is Sum sumA)
        {
            var newTerms = new List<Expr>(sumA.Terms.Count + 1);
            newTerms.AddRange(sumA.Terms);
            newTerms.Add(new Constant(b));
            return new Sum(newTerms);
        }
        return new Sum([a, new Constant(b)]);
    }

    public static Expr operator +(double a, Expr b)
    {
        if (b is Sum sumB)
        {
            var newTerms = new List<Expr>(sumB.Terms.Count + 1);
            newTerms.Add(new Constant(a));
            newTerms.AddRange(sumB.Terms);
            return new Sum(newTerms);
        }
        return new Sum([new Constant(a), b]);
    }
    public static Expr operator -(Expr a, double b)
    {
        if (a is Sum sumA)
        {
            var newTerms = new List<Expr>(sumA.Terms.Count + 1);
            newTerms.AddRange(sumA.Terms);
            newTerms.Add(new Constant(-b));
            return new Sum(newTerms);
        }
        return new Sum([a, new Constant(-b)]);
    }

    public static Expr operator -(double a, Expr b)
    {
        return new Sum([new Constant(a), -b]);
    }
    public static Expr operator *(Expr a, double b)
    {
        if (a is Product prodA)
        {
            var newFactors = new List<Expr>(prodA.Factors.Count + 1);
            newFactors.AddRange(prodA.Factors);
            newFactors.Add(new Constant(b));
            return new Product(newFactors);
        }
        return new Product([a, new Constant(b)]);
    }

    public static Expr operator *(double a, Expr b)
    {
        if (b is Product prodB)
        {
            var newFactors = new List<Expr>(prodB.Factors.Count + 1);
            newFactors.Add(new Constant(a));
            newFactors.AddRange(prodB.Factors);
            return new Product(newFactors);
        }
        return new Product([new Constant(a), b]);
    }
    public static Expr operator /(Expr a, double b) => a * (1.0 / b);
    public static Expr operator /(double a, Expr b) => new Division(new Constant(a), b);

    // C# 14 compound assignment operators - modify expression in-place for efficiency
    public void operator +=(Expr other)
    {
        // Check if we've been replaced with a Sum
        if (_replacement is Sum sum)
            sum.Terms.Add(other);
        // Check if this is a zero constant with no replacement yet
        else if (_replacement is null && this is Constant { Value: 0 })
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
        else if (_replacement is null && this is Constant { Value: 0 })
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
        else if (_replacement is null && this is Constant { Value: 1 })
            ReplaceWith(other);
        // Otherwise create a new Product with current value and new factor
        else
            ReplaceWith(new Product([Clone(), other]));
    }

    public void operator /=(Expr other)
    {
        // Check if we've been replaced with a Product (add reciprocal)
        if (_replacement is Product product)
            product.Factors.Add(new Division(1, other));
        // Otherwise create a division
        else
            ReplaceWith(new Division(Clone(), other));
    }

    public Constraint Between(double lower, double upper) => new Constraint(this, lower, upper);

    protected Expr Clone()
    {
        if (_replacement is not null)
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

    public static Constraint operator >=(double value, Expr expr) => new(expr, double.NegativeInfinity, value);
    public static Constraint operator <=(double value, Expr expr) => new(expr, value, double.PositiveInfinity);
    public static Constraint operator ==(double value, Expr expr) => new(expr, value, value);
    public static Constraint operator !=(double value, Expr expr) => throw new NotSupportedException("Inequality constraints (!=) are not supported in optimization models.");

    public static Constraint operator >=(Expr left, Expr right) => new(left - right, 0, double.PositiveInfinity);
    public static Constraint operator <=(Expr left, Expr right) => new(left - right, double.NegativeInfinity, 0);
    public static Constraint operator ==(Expr left, Expr right) => new(left - right, 0, 0);
    public static Constraint operator !=(Expr left, Expr right) => throw new NotSupportedException("Inequality constraints (!=) are not supported in optimization models.");

    public static Expr Pow(Expr @base, double exponent) => new PowerOp(@base, exponent);
    public static Expr Pow(Expr @base, Expr exponent) => Exp(exponent * Log(@base));
    public static Expr Sqrt(Expr a) => new PowerOp(a, 0.5);
    public static Expr Sin(Expr a) => new Sin(a);
    public static Expr Cos(Expr a) => new Cos(a);
    public static Expr Tan(Expr a) => new Tan(a);
    public static Expr Exp(Expr a) => new Exp(a);
    public static Expr Log(Expr a) => new Log(a);
}

public sealed class Constant : Expr
{
    public double Value { get; set; }

    public Constant(double value) => Value = value;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Value;
    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier) { }
    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier) { }
    protected override void CollectVariablesCore(HashSet<Variable> variables) { }
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries) { }
    protected override bool IsConstantWrtXCore() => true;
    protected override bool IsLinearCore() => true;
    protected override bool IsAtMostQuadraticCore() => true;

    protected override Expr CloneCore() => new Constant(Value);
}

public sealed class Division : Expr
{
    public Expr Left { get; set; }
    public Expr Right { get; set; }

    public Division(Expr left, Expr right)
    {
        Left = left;
        Right = right;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var l = Left.Evaluate(x);
        var r = Right.Evaluate(x);
        return l / r;
    }

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        // d(L/R)/dx = (dL/dx * R - L * dR/dx) / R²
        var rVal = Right.Evaluate(x);
        var lVal = Left.Evaluate(x);
        Left.AccumulateGradient(x, grad, multiplier / rVal);
        Right.AccumulateGradient(x, grad, -multiplier * lVal / (rVal * rVal));
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        // d(L/R)/dx = (dL/dx * R - L * dR/dx) / R²
        // d²(L/R)/dx² = d²L/dx²/R - L*d²R/dx²/R² - 2*dL/dx*dR/dx/R² + 2*L*(dR/dx)²/R³
        var rVal = Right.Evaluate(x);
        var lVal = Left.Evaluate(x);
        var r2 = rVal * rVal;
        var r3 = r2 * rVal;
        Left.AccumulateHessian(x, grad, hess, multiplier / rVal);
        Right.AccumulateHessian(x, grad, hess, -multiplier * lVal / r2);
        var n = grad.Length;
        Span<double> gradL = new double[n];
        Span<double> gradR = new double[n];
        Left.AccumulateGradient(x, gradL, 1.0);
        Right.AccumulateGradient(x, gradR, 1.0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                var cross = -multiplier / r2 * (gradL[i] * gradR[j] + gradL[j] * gradR[i]);
                var rr = multiplier * 2 * lVal / r3 * gradR[i] * gradR[j];
                hess.Add(i, j, cross + rr);
            }
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        Left.CollectVariables(variables);
        Right.CollectVariables(variables);
    }

    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        if (Right.IsConstantWrtX())
        {
            Left.CollectHessianSparsity(entries);
        }
        else
        {
            var vars = new HashSet<Variable>();
            Left.CollectVariables(vars);
            Right.CollectVariables(vars);
            AddClique(entries, vars);
        }
    }

    protected override bool IsConstantWrtXCore() => Left.IsConstantWrtX() && Right.IsConstantWrtX();

    protected override bool IsLinearCore()
    {
        // Linear if numerator is linear and denominator is constant
        return Left.IsLinear() && Right.IsConstantWrtX();
    }

    protected override bool IsAtMostQuadraticCore()
    {
        // At most quadratic if numerator is at most quadratic and denominator is constant
        return Left.IsAtMostQuadratic() && Right.IsConstantWrtX();
    }

    protected override Expr CloneCore() => new Division(Left, Right);
}

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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        Operand.AccumulateHessian(x, grad, hess, -multiplier);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Operand.CollectVariables(variables);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries) => Operand.CollectHessianSparsity(entries);
    protected override bool IsConstantWrtXCore() => Operand.IsConstantWrtX();
    protected override bool IsLinearCore() => Operand.IsLinear();
    protected override bool IsAtMostQuadraticCore() => Operand.IsAtMostQuadratic();

    protected override Expr CloneCore() => new Negation(Operand);
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
        Span<double> gradB = new double[n];
        Base.AccumulateGradient(x, gradB, 1.0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                hess.Add(i, j, multiplier * secondDerivCoeff * gradB[i] * gradB[j]);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Base.CollectVariables(variables);

    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        if (Math.Abs(Exponent - 1.0) < 1e-15)
        {
            Base.CollectHessianSparsity(entries);
        }
        else if (!Base.IsConstantWrtX())
        {
            var vars = new HashSet<Variable>();
            Base.CollectVariables(vars);
            AddClique(entries, vars);
        }
    }

    protected override bool IsConstantWrtXCore() => Base.IsConstantWrtX();

    protected override bool IsLinearCore()
    {
        // Linear if exponent is 1 and base is linear, or if base is constant
        return (Math.Abs(Exponent - 1.0) < 1e-15 && Base.IsLinear()) || Base.IsConstantWrtX();
    }

    protected override bool IsAtMostQuadraticCore()
    {
        // At most quadratic if: (exponent is at most 2 and base is linear) or (exponent is 1 and base is quadratic) or (base is constant)
        if (Base.IsConstantWrtX()) return true;
        if (Math.Abs(Exponent - 1.0) < 1e-15) return Base.IsAtMostQuadratic();
        if (Math.Abs(Exponent - 2.0) < 1e-15) return Base.IsLinear();
        return false;
    }

    protected override Expr CloneCore() => new PowerOp(Base, Exponent);
}

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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, grad, hess, multiplier * Math.Cos(arg));
        var n = grad.Length;
        Span<double> gradArg = new double[n];
        Argument.AccumulateGradient(x, gradArg, 1.0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                hess.Add(i, j, multiplier * -Math.Sin(arg) * gradArg[i] * gradArg[j]);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Argument.CollectVariables(variables);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        if (!Argument.IsConstantWrtX())
        {
            var vars = new HashSet<Variable>();
            Argument.CollectVariables(vars);
            AddClique(entries, vars);
        }
    }
    protected override bool IsConstantWrtXCore() => Argument.IsConstantWrtX();
    protected override bool IsLinearCore() => Argument.IsConstantWrtX();
    protected override bool IsAtMostQuadraticCore() => Argument.IsConstantWrtX();

    protected override Expr CloneCore() => new Sin(Argument);
}

public sealed class Cos : Expr
{
    public Expr Argument { get; set; }

    public Cos(Expr argument) => Argument = argument;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Cos(Argument.Evaluate(x));

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateGradient(x, grad, multiplier * -Math.Sin(arg));
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, grad, hess, multiplier * -Math.Sin(arg));
        var n = grad.Length;
        Span<double> gradArg = new double[n];
        Argument.AccumulateGradient(x, gradArg, 1.0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                hess.Add(i, j, multiplier * -Math.Cos(arg) * gradArg[i] * gradArg[j]);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Argument.CollectVariables(variables);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        if (!Argument.IsConstantWrtX())
        {
            var vars = new HashSet<Variable>();
            Argument.CollectVariables(vars);
            AddClique(entries, vars);
        }
    }
    protected override bool IsConstantWrtXCore() => Argument.IsConstantWrtX();
    protected override bool IsLinearCore() => Argument.IsConstantWrtX();
    protected override bool IsAtMostQuadraticCore() => Argument.IsConstantWrtX();

    protected override Expr CloneCore() => new Cos(Argument);
}

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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        var cos = Math.Cos(arg);
        Argument.AccumulateHessian(x, grad, hess, multiplier / (cos * cos));
        var n = grad.Length;
        Span<double> gradArg = new double[n];
        Argument.AccumulateGradient(x, gradArg, 1.0);
        var secondDeriv = 2 * Math.Tan(arg) / (cos * cos);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                hess.Add(i, j, multiplier * secondDeriv * gradArg[i] * gradArg[j]);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Argument.CollectVariables(variables);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        if (!Argument.IsConstantWrtX())
        {
            var vars = new HashSet<Variable>();
            Argument.CollectVariables(vars);
            AddClique(entries, vars);
        }
    }
    protected override bool IsConstantWrtXCore() => Argument.IsConstantWrtX();
    protected override bool IsLinearCore() => Argument.IsConstantWrtX();
    protected override bool IsAtMostQuadraticCore() => Argument.IsConstantWrtX();

    protected override Expr CloneCore() => new Tan(Argument);
}

public sealed class Exp : Expr
{
    public Expr Argument { get; set; }

    public Exp(Expr argument) => Argument = argument;

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Exp(Argument.Evaluate(x));

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateGradient(x, grad, multiplier * Math.Exp(arg));
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        var expVal = Math.Exp(arg);
        Argument.AccumulateHessian(x, grad, hess, multiplier * expVal);
        var n = grad.Length;
        Span<double> gradArg = new double[n];
        Argument.AccumulateGradient(x, gradArg, 1.0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                hess.Add(i, j, multiplier * expVal * gradArg[i] * gradArg[j]);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Argument.CollectVariables(variables);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        if (!Argument.IsConstantWrtX())
        {
            var vars = new HashSet<Variable>();
            Argument.CollectVariables(vars);
            AddClique(entries, vars);
        }
    }
    protected override bool IsConstantWrtXCore() => Argument.IsConstantWrtX();
    protected override bool IsLinearCore() => Argument.IsConstantWrtX();
    protected override bool IsAtMostQuadraticCore() => Argument.IsConstantWrtX();

    protected override Expr CloneCore() => new Exp(Argument);
}

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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, grad, hess, multiplier / arg);
        var n = grad.Length;
        Span<double> gradArg = new double[n];
        Argument.AccumulateGradient(x, gradArg, 1.0);
        var secondDeriv = -1.0 / (arg * arg);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                hess.Add(i, j, multiplier * secondDeriv * gradArg[i] * gradArg[j]);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables) => Argument.CollectVariables(variables);
    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        if (!Argument.IsConstantWrtX())
        {
            var vars = new HashSet<Variable>();
            Argument.CollectVariables(vars);
            AddClique(entries, vars);
        }
    }
    protected override bool IsConstantWrtXCore() => Argument.IsConstantWrtX();
    protected override bool IsLinearCore() => Argument.IsConstantWrtX();
    protected override bool IsAtMostQuadraticCore() => Argument.IsConstantWrtX();

    protected override Expr CloneCore() => new Log(Argument);
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

    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        foreach (var term in Terms)
            term.CollectHessianSparsity(entries);
    }

    protected override bool IsConstantWrtXCore() => Terms.All(t => t.IsConstantWrtX());
    protected override bool IsLinearCore() => Terms.All(t => t.IsLinear());
    protected override bool IsAtMostQuadraticCore() => Terms.All(t => t.IsAtMostQuadratic());

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
        for (int i = 0; i < Factors.Count; i++)
        {
            var otherProduct = 1.0;
            for (int j = 0; j < Factors.Count; j++)
            {
                if (i != j)
                    otherProduct *= Factors[j].Evaluate(x);
            }
            Factors[i].AccumulateGradient(x, grad, multiplier * otherProduct);
        }
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, Span<double> grad, HessianAccumulator hess, double multiplier)
    {
        if (Factors.Count == 0)
            return;
        if (Factors.Count == 1)
        {
            Factors[0].AccumulateHessian(x, grad, hess, multiplier);
            return;
        }

        var n = grad.Length;

        // Evaluate all factors once
        var factorValues = new double[Factors.Count];
        for (int i = 0; i < Factors.Count; i++)
            factorValues[i] = Factors[i].Evaluate(x);

        // Compute gradients of all factors once and keep track of non-zeros
        var factorGradients = new double[Factors.Count][];
        var nonZeroIndices = new List<int>[Factors.Count];

        for (int i = 0; i < Factors.Count; i++)
        {
            factorGradients[i] = new double[n];
            Factors[i].AccumulateGradient(x, factorGradients[i], 1.0);

            var nonZeros = new List<int>();
            for (int j = 0; j < n; j++)
            {
                if (Math.Abs(factorGradients[i][j]) > 1e-18)
                    nonZeros.Add(j);
            }
            nonZeroIndices[i] = nonZeros;
        }

        // 1. Accumulate Hessian from each factor's second derivative
        //    d²(f₁*f₂*...*fₙ)/dx² includes: d²fₖ/dx² * (∏ᵢ≠ₖ fᵢ)
        for (int k = 0; k < Factors.Count; k++)
        {
            var otherProduct = 1.0;
            for (int l = 0; l < Factors.Count; l++)
            {
                if (l != k)
                {
                    otherProduct *= factorValues[l];
                    if (Math.Abs(otherProduct) < 1e-100) break; // Optimization: bail out if product becomes zero
                }
            }

            if (Math.Abs(otherProduct) > 1e-18)
                Factors[k].AccumulateHessian(x, grad, hess, multiplier * otherProduct);
        }

        // 2. Add cross terms between pairs of factors
        //    d²(f₁*f₂*...*fₙ)/dx² includes: ∂fₖ/∂xᵢ * ∂fₘ/∂xⱼ * (∏ₗ≠ₖ,ₘ fₗ)
        for (int k = 0; k < Factors.Count; k++)
        {
            if (nonZeroIndices[k].Count == 0) continue;

            for (int m = k + 1; m < Factors.Count; m++)
            {
                if (nonZeroIndices[m].Count == 0) continue;

                var otherProduct = 1.0;
                for (int l = 0; l < Factors.Count; l++)
                {
                    if (l != k && l != m)
                    {
                        otherProduct *= factorValues[l];
                        if (Math.Abs(otherProduct) < 1e-100) break;
                    }
                }

                if (Math.Abs(otherProduct) > 1e-18)
                {
                    var coeff = multiplier * otherProduct;
                    var idxK = nonZeroIndices[k];
                    var idxM = nonZeroIndices[m];
                    var gradK = factorGradients[k];
                    var gradM = factorGradients[m];

                    foreach (var i in idxK)
                    {
                        foreach (var j in idxM)
                        {
                            // H_ij += (gK_i * gM_j + gK_j * gM_i) * coeff
                            // We call Add twice to cover both symmetric terms
                            hess.Add(i, j, coeff * gradK[i] * gradM[j]);
                            hess.Add(j, i, coeff * gradK[j] * gradM[i]);
                        }
                    }
                }
            }
        }
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        foreach (var factor in Factors)
            factor.CollectVariables(variables);
    }

    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        foreach (var factor in Factors)
            factor.CollectHessianSparsity(entries);

        var factorVars = new HashSet<Variable>[Factors.Count];
        for (int i = 0; i < Factors.Count; i++)
        {
            factorVars[i] = new HashSet<Variable>();
            Factors[i].CollectVariables(factorVars[i]);
        }

        for (int i = 0; i < Factors.Count; i++)
        {
            for (int j = i + 1; j < Factors.Count; j++)
            {
                foreach (var v1 in factorVars[i])
                    foreach (var v2 in factorVars[j])
                        AddSparsityEntry(entries, v1.Index, v2.Index);
            }
        }
    }

    protected override bool IsConstantWrtXCore() => Factors.All(f => f.IsConstantWrtX());

    protected override bool IsLinearCore()
    {
        // Linear if at most one factor is non-constant and that factor is linear
        var nonConstantFactors = Factors.Where(f => !f.IsConstantWrtX()).ToList();
        return nonConstantFactors.Count == 0 || (nonConstantFactors.Count == 1 && nonConstantFactors[0].IsLinear());
    }

    protected override bool IsAtMostQuadraticCore()
    {
        // Count non-constant factors and their degrees
        var nonConstantFactors = Factors.Where(f => !f.IsConstantWrtX()).ToList();

        if (nonConstantFactors.Count == 0)
            return true; // All constant

        if (nonConstantFactors.Count == 1)
            return nonConstantFactors[0].IsAtMostQuadratic(); // One factor, check if it's at most quadratic

        if (nonConstantFactors.Count == 2)
            return nonConstantFactors.All(f => f.IsLinear()); // Two linear factors: degree 1*1 = 2

        return false; // More than two non-constant factors means degree > 2
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
