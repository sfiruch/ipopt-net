using System.Runtime.CompilerServices;
using System.Buffers;
using System.Runtime.InteropServices;

namespace IpoptNet.Modelling;

public abstract class Expr
{
    protected Expr? _replacement;

    public double Evaluate(ReadOnlySpan<double> x) => _replacement?.Evaluate(x) ?? EvaluateCore(x);
    public void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad, double multiplier) =>
        (_replacement ?? this).AccumulateGradientCore(x, grad, multiplier);
    public void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier) =>
        (_replacement ?? this).AccumulateHessianCore(x, hess, multiplier);
    public void CollectVariables(HashSet<Variable> variables) =>
        (_replacement ?? this).CollectVariablesCore(variables);
    public void CollectHessianSparsity(HashSet<(int row, int col)> entries) =>
        (_replacement ?? this).CollectHessianSparsityCore(entries);

    protected void AccumulateOuterHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double coeff, HashSet<Variable> vars, double[] rentedGrad)
    {
        if (Math.Abs(coeff) < 1e-18) return;

        // Only iterate over non-zero entries
        var nonZeros = new List<int>(vars.Count);
        foreach (var v in vars)
            if (Math.Abs(rentedGrad[v.Index]) > 1e-18)
                nonZeros.Add(v.Index);

        nonZeros.Sort();

        for (int i = 0; i < nonZeros.Count; i++)
        {
            int row = nonZeros[i];
            double valI = rentedGrad[row];
            for (int j = 0; j <= i; j++)
            {
                int col = nonZeros[j];
                hess.Add(row, col, coeff * valI * rentedGrad[col]);
            }
        }
    }

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
    protected abstract void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier);
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
    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier) { }
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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // f = l / r
        // df/dx = (l'r - lr') / r²
        // d²f/dx² = ((l''r + l'r' - l'r' - lr'')r² - (l'r - lr')2rr') / r⁴
        //         = ((l''r - lr'')r² - 2rr'(l'r - lr')) / r⁴
        //         = (l''r - lr'')/r² - 2r'(l'r - lr')/r³
        //         = l''/r - lr''/r² - 2l'r'/r² + 2lr'²/r³
        var lVal = Left.Evaluate(x);
        var rVal = Right.Evaluate(x);
        var r2 = rVal * rVal;
        var r3 = r2 * rVal;

        if (Math.Abs(multiplier) < 1e-18) return;

        Left.AccumulateHessian(x, hess, multiplier / rVal);
        Right.AccumulateHessian(x, hess, -multiplier * lVal / r2);

        var n = x.Length;
        var gradL = ArrayPool<double>.Shared.Rent(n);
        var gradR = ArrayPool<double>.Shared.Rent(n);

        var varsL = new HashSet<Variable>();
        var varsR = new HashSet<Variable>();
        Left.CollectVariables(varsL);
        Right.CollectVariables(varsR);

        if (varsL.Count < n / 32) foreach (var v in varsL) gradL[v.Index] = 0; else Array.Clear(gradL);
        if (varsR.Count < n / 32) foreach (var v in varsR) gradR[v.Index] = 0; else Array.Clear(gradR);

        Left.AccumulateGradient(x, gradL, 1.0);
        Right.AccumulateGradient(x, gradR, 1.0);

        // Add 2lr'²/r³ (outer product of r' with itself)
        AccumulateOuterHessian(x, hess, multiplier * 2 * lVal / r3, varsR, gradR);

        // Add -l'r'/r² (outer products between l' and r')
        // Cross derivative term is -(l'_x r'_y + l'_y r'_x) / r²
        var coeff = -multiplier / r2;
        var nonZerosL = new List<int>(varsL.Count);
        foreach (var v in varsL) if (Math.Abs(gradL[v.Index]) > 1e-18) nonZerosL.Add(v.Index);
        var nonZerosR = new List<int>(varsR.Count);
        foreach (var v in varsR) if (Math.Abs(gradR[v.Index]) > 1e-18) nonZerosR.Add(v.Index);

        foreach (var i in nonZerosL)
        {
            var valLi = gradL[i];
            foreach (var j in nonZerosR)
            {
                // Symmetric contribution: -(l'_i * r'_j + l'_j * r'_i) / r²
                // We add both directions to ensure symmetry if the index sets overlap.
                // For x/y, l'=[1,0], r'=[0,1]:
                // i=0, j=1: Adds coeff * l'[0] * r'[1] = coeff * 1 * 1
                //           Adds coeff * l'[1] * r'[0] = coeff * 0 * 0
                // Total H[1,0] = coeff
                hess.Add(i, j, coeff * valLi * gradR[j]);
                hess.Add(j, i, coeff * gradL[j] * gradR[i]);
            }
        }

        ArrayPool<double>.Shared.Return(gradL);
        ArrayPool<double>.Shared.Return(gradR);
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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // d²(b^n)/dx² = n*(n-1)*b^(n-2)*(db/dx)² + n*b^(n-1)*d²b/dx²
        var bVal = Base.Evaluate(x);
        var firstDerivCoeff = Exponent * Math.Pow(bVal, Exponent - 1);
        var secondDerivCoeff = Exponent * (Exponent - 1) * Math.Pow(bVal, Exponent - 2);
        Base.AccumulateHessian(x, hess, multiplier * firstDerivCoeff);

        var n = x.Length;
        var gradB = ArrayPool<double>.Shared.Rent(n);
        
        var vars = new HashSet<Variable>();
        Base.CollectVariables(vars);

        if (vars.Count < n / 32)
            foreach (var v in vars) gradB[v.Index] = 0;
        else
            Array.Clear(gradB);

        Base.AccumulateGradient(x, gradB, 1.0);
        AccumulateOuterHessian(x, hess, multiplier * secondDerivCoeff, vars, gradB);

        ArrayPool<double>.Shared.Return(gradB);
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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, hess, multiplier * Math.Cos(arg));

        var n = x.Length;
        var gradArg = ArrayPool<double>.Shared.Rent(n);
        
        var vars = new HashSet<Variable>();
        Argument.CollectVariables(vars);

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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, hess, multiplier * -Math.Sin(arg));

        var n = x.Length;
        var gradArg = ArrayPool<double>.Shared.Rent(n);
        
        var vars = new HashSet<Variable>();
        Argument.CollectVariables(vars);

        if (vars.Count < n / 32)
            foreach (var v in vars) gradArg[v.Index] = 0;
        else
            Array.Clear(gradArg);

        Argument.AccumulateGradient(x, gradArg, 1.0);
        AccumulateOuterHessian(x, hess, multiplier * -Math.Cos(arg), vars, gradArg);

        ArrayPool<double>.Shared.Return(gradArg);
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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        var cos = Math.Cos(arg);
        Argument.AccumulateHessian(x, hess, multiplier / (cos * cos));

        var n = x.Length;
        var gradArg = ArrayPool<double>.Shared.Rent(n);
        
        var vars = new HashSet<Variable>();
        Argument.CollectVariables(vars);

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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        var expVal = Math.Exp(arg);
        Argument.AccumulateHessian(x, hess, multiplier * expVal);

        var n = x.Length;
        var gradArg = ArrayPool<double>.Shared.Rent(n);
        
        var vars = new HashSet<Variable>();
        Argument.CollectVariables(vars);

        if (vars.Count < n / 32)
            foreach (var v in vars) gradArg[v.Index] = 0;
        else
            Array.Clear(gradArg);

        Argument.AccumulateGradient(x, gradArg, 1.0);
        AccumulateOuterHessian(x, hess, multiplier * expVal, vars, gradArg);

        ArrayPool<double>.Shared.Return(gradArg);
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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var arg = Argument.Evaluate(x);
        Argument.AccumulateHessian(x, hess, multiplier / arg);

        var n = x.Length;
        var gradArg = ArrayPool<double>.Shared.Rent(n);
        
        var vars = new HashSet<Variable>();
        Argument.CollectVariables(vars);

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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        foreach (var term in Terms)
            term.AccumulateHessian(x, hess, multiplier);
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

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        if (Factors.Count == 0)
            return;
        if (Factors.Count == 1)
        {
            Factors[0].AccumulateHessian(x, hess, multiplier);
            return;
        }

        var n = x.Length;

        // Evaluate all factors once
        var factorValues = new double[Factors.Count];
        for (int i = 0; i < Factors.Count; i++)
            factorValues[i] = Factors[i].Evaluate(x);

        // Compute gradients of all factors once and keep track of non-zeros
        var factorGradients = new double[Factors.Count][];
        var nonZeroIndices = new List<int>[Factors.Count];

        for (int i = 0; i < Factors.Count; i++)
        {
            if (Factors[i].IsConstantWrtX())
            {
                nonZeroIndices[i] = [];
                continue;
            }

            // 1. Identify variables involved first
            var vars = new HashSet<Variable>();
            Factors[i].CollectVariables(vars);

            var rented = ArrayPool<double>.Shared.Rent(n);
            
            // 2. Zero-init ONLY involved variables if sparse
            if (vars.Count < n / 32)
            {
                foreach (var v in vars)
                    rented[v.Index] = 0.0;
            }
            else
            {
                Array.Clear(rented);
            }

            factorGradients[i] = rented;
            Factors[i].AccumulateGradient(x, rented, 1.0);

            // 3. Identify non-zeros
            var nonZeros = new List<int>(vars.Count);
            foreach (var v in vars)
            {
                if (Math.Abs(rented[v.Index]) > 1e-18)
                    nonZeros.Add(v.Index);
            }
            nonZeroIndices[i] = nonZeros;
        }

        // 1. Accumulate Hessian from each factor's second derivative
        for (int k = 0; k < Factors.Count; k++)
        {
            var otherProduct = 1.0;
            for (int l = 0; l < Factors.Count; l++)
            {
                if (l != k)
                {
                    otherProduct *= factorValues[l];
                    if (Math.Abs(otherProduct) < 1e-100) break;
                }
            }

            if (Math.Abs(otherProduct) > 1e-18)
                Factors[k].AccumulateHessian(x, hess, multiplier * otherProduct);
        }

        // 2. Add cross terms between pairs of factors
        for (int k = 0; k < Factors.Count; k++)
        {
            var idxK = nonZeroIndices[k];
            if (idxK.Count == 0) continue;
            var gradK = factorGradients[k];
            var spanK = CollectionsMarshal.AsSpan(idxK);

            for (int m = k + 1; m < Factors.Count; m++)
            {
                var idxM = nonZeroIndices[m];
                if (idxM.Count == 0) continue;

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
                    var gradM = factorGradients[m];
                    var spanM = CollectionsMarshal.AsSpan(idxM);

                    // CROSS-TERM HESSIAN COMPUTATION (Product Rule)
                    // =============================================
                    // For product f = F_k * F_m (where other factors are constant),
                    // the Hessian cross-terms are:
                    //   H = (∇F_k)(∇F_m)^T + (∇F_m)(∇F_k)^T
                    //
                    // This is a SYMMETRIC outer product. We compute BOTH parts:
                    //   H[i,j] += coeff * gradK[i] * gradM[j]  (first outer product)
                    //   H[j,i] += coeff * gradM[j] * gradK[i]  (transpose)
                    //
                    // Since hess.Add() normalizes to lower-triangular form,
                    // BOTH calls may map to the SAME entry (i,j) when swapped.
                    //
                    // IMPORTANT: This is NOT double-counting! Here's why:
                    // - When i and j are from DIFFERENT factor gradient index sets,
                    //   one of gradK[j] or gradM[i] is typically zero
                    // - Example: For f = x*y*z, cross-term between x and y:
                    //     gradK = [1,0,0] (gradient of x)
                    //     gradM = [0,1,0] (gradient of y)
                    //     At (i=0, j=1): gradK[0]*gradM[1] = 1*1 = 1 ✓
                    //                    gradM[0]*gradK[1] = 0*0 = 0 ✓
                    //     Total contribution = 1 (correct!)
                    //
                    // The two hess.Add() calls ensure we capture BOTH parts of
                    // the symmetric outer product correctly.
                    foreach (var i in spanK)
                    {
                        var gKi = gradK[i];
                        var gMi = gradM[i]; 
                        var c_gKi = coeff * gKi;
                        var c_gMi = coeff * gMi;

                        foreach (var j in spanM)
                        {
                            var gMj = gradM[j];
                            var gKj = gradK[j]; 
                            
                            // Add both parts of the symmetric outer product
                            hess.Add(i, j, c_gKi * gMj);
                            hess.Add(j, i, c_gMi * gKj);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < Factors.Count; i++)
            if (factorGradients[i] != null)
                ArrayPool<double>.Shared.Return(factorGradients[i]);
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
    private readonly Dictionary<long, double> _entries = new();
    private readonly int _n;

    public HessianAccumulator(int n) => _n = n;

    public void Add(int i, int j, double value)
    {
        if (Math.Abs(value) < 1e-18) return;
        
        long key = i >= j ? ((long)i << 32) | (uint)j : ((long)j << 32) | (uint)i;
        ref var entry = ref CollectionsMarshal.GetValueRefOrAddDefault(_entries, key, out _);
        entry += value;
    }

    public void Clear() => _entries.Clear();

    public int Count => _entries.Count;

    public bool TryGetValue(int i, int j, out double value)
    {
        long key = i >= j ? ((long)i << 32) | (uint)j : ((long)j << 32) | (uint)i;
        return _entries.TryGetValue(key, out value);
    }

    public IReadOnlyDictionary<(int row, int col), double> Entries =>
        _entries.ToDictionary(kvp => ((int)(kvp.Key >> 32), (int)(kvp.Key & 0xFFFFFFFF)), kvp => kvp.Value);

    public IEnumerable<((int row, int col) key, double value)> GetEntries()
    {
        foreach (var kvp in _entries)
        {
            yield return (((int)(kvp.Key >> 32), (int)(kvp.Key & 0xFFFFFFFF)), kvp.Value);
        }
    }

    public int GetNonZeroCount()
    {
        // For IPOPT, we need the structure of the lower triangular Hessian
        var count = 0;
        foreach (var entry in _entries)
            if (Math.Abs(entry.Value) >= 1e-18)
                count++;
        return count;
    }
}
