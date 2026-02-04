using System.Runtime.CompilerServices;
using System.Buffers;
using System.Runtime.InteropServices;

namespace IpoptNet.Modelling;

public abstract class Expr
{
    protected Expr? _replacement;
    
    /// <summary>
    /// Gets the actual expression, following any replacements. For testing purposes.
    /// </summary>
    public Expr GetActual() => _replacement ?? this;

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
        // If either operand is QuadExpr, result should be QuadExpr
        // BUT only if the other operand is also at most quadratic
        if (a is QuadExpr && b.IsAtMostQuadratic())
        {
            return new QuadExpr([a, b]);
        }
        if (b is QuadExpr && a.IsAtMostQuadratic())
        {
            return new QuadExpr([a, b]);
        }
        
        // Auto-create QuadExpr if BOTH operands are at most quadratic
        // and at least one is actually quadratic (non-linear)
        bool aIsQuadratic = !a.IsLinear() && a.IsAtMostQuadratic();
        bool bIsQuadratic = !b.IsLinear() && b.IsAtMostQuadratic();
        
        if ((aIsQuadratic || bIsQuadratic) && a.IsAtMostQuadratic() && b.IsAtMostQuadratic())
        {
            return new QuadExpr([a, b]);
        }
        
        // Always create a new LinExpr to ensure proper merging
        return new LinExpr([a, b]);
    }
    
    public static Expr operator -(Expr a, Expr b)
    {
        // If either operand is QuadExpr, result should be QuadExpr
        // BUT only if the other operand is also at most quadratic
        if (a is QuadExpr && b.IsAtMostQuadratic())
        {
            return new QuadExpr([a, -b]);
        }
        if (b is QuadExpr && a.IsAtMostQuadratic())
        {
            return new QuadExpr([a, -b]);
        }
        
        // Auto-create QuadExpr if BOTH operands are at most quadratic
        bool aIsQuadratic = !a.IsLinear() && a.IsAtMostQuadratic();
        bool bIsQuadratic = !b.IsLinear() && b.IsAtMostQuadratic();
        
        if ((aIsQuadratic || bIsQuadratic) && a.IsAtMostQuadratic() && b.IsAtMostQuadratic())
        {
            return new QuadExpr([a, -b]);
        }
        
        // Always create a new LinExpr to ensure proper merging
        return new LinExpr([a, -b]);
    }
    public static Expr operator *(Expr a, Expr b)
    {
        // If left operand is already a Product, extend it with the right operand
        if (a is Product prodA)
        {
            return new Product([.. prodA.Factors, b]);
        }
        // If right operand is a Product, prepend the left operand to it
        else if (b is Product prodB)
        {
            return new Product([a, .. prodB.Factors]);
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
        // Preserve QuadExpr if present
        if (a is QuadExpr)
        {
            return new QuadExpr([a, new Constant(b)]);
        }
        // Auto-create QuadExpr for quadratic expressions
        if (!a.IsLinear() && a.IsAtMostQuadratic())
        {
            return new QuadExpr([a, new Constant(b)]);
        }
        return new LinExpr([a, new Constant(b)]);
    }

    public static Expr operator +(double a, Expr b)
    {
        // Preserve QuadExpr if present
        if (b is QuadExpr)
        {
            return new QuadExpr([new Constant(a), b]);
        }
        // Auto-create QuadExpr for quadratic expressions
        if (!b.IsLinear() && b.IsAtMostQuadratic())
        {
            return new QuadExpr([new Constant(a), b]);
        }
        return new LinExpr([new Constant(a), b]);
    }
    public static Expr operator -(Expr a, double b)
    {
        // Preserve QuadExpr if present
        if (a is QuadExpr)
        {
            return new QuadExpr([a, new Constant(-b)]);
        }
        // Auto-create QuadExpr for quadratic expressions
        if (!a.IsLinear() && a.IsAtMostQuadratic())
        {
            return new QuadExpr([a, new Constant(-b)]);
        }
        return new LinExpr([a, new Constant(-b)]);
    }

    public static Expr operator -(double a, Expr b)
    {
        // Preserve QuadExpr if present
        if (b is QuadExpr)
        {
            return new QuadExpr([new Constant(a), -b]);
        }
        // Auto-create QuadExpr for quadratic expressions
        if (!b.IsLinear() && b.IsAtMostQuadratic())
        {
            return new QuadExpr([new Constant(a), -b]);
        }
        return new LinExpr([new Constant(a), -b]);
    }
    public static Expr operator *(Expr a, double b)
    {
        if (a is Product prodA)
        {
            return new Product([.. prodA.Factors, new Constant(b)]);
        }
        return new Product([a, new Constant(b)]);
    }

    public static Expr operator *(double a, Expr b)
    {
        if (b is Product prodB)
        {
            return new Product([new Constant(a), .. prodB.Factors]);
        }
        return new Product([new Constant(a), b]);
    }
    public static Expr operator /(Expr a, double b) => a * (1.0 / b);
    public static Expr operator /(double a, Expr b) => new Division(new Constant(a), b);

    // C# 14 compound assignment operators - modify expression in-place for efficiency
    public void operator +=(Expr other)
    {
        // Special case: if this is a zero constant, just replace with other
        if (_replacement is null && this is Constant { Value: 0 })
        {
            ReplaceWith(other);
            return;
        }
        
        // Always use the proper + operator to ensure correct processing
        // The + operator will create a new QuadExpr/LinExpr with proper term extraction
        var result = Clone() + other;
        ReplaceWith(result);
    }

    public void operator -=(Expr other)
    {
        // Special case: if this is a zero constant, just replace with -other
        if (_replacement is null && this is Constant { Value: 0 })
        {
            ReplaceWith(-other);
            return;
        }
        
        // Always use the proper - operator to ensure correct processing
        var result = Clone() - other;
        ReplaceWith(result);
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

    /// <summary>
    /// Prints the expression tree for debugging.
    /// </summary>
    /// <param name="writer">The TextWriter to write to. Defaults to Console.Out.</param>
    /// <param name="indent">The indentation string for formatting.</param>
    public void Print(TextWriter? writer = null, string indent = "")
    {
        writer ??= Console.Out;
        if (_replacement is not null)
        {
            writer.WriteLine($"{indent}[Replaced with:]");
            _replacement.Print(writer, indent + "  ");
            return;
        }
        PrintCore(writer, indent);
    }

    protected virtual void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}{GetType().Name}");
    }
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

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}Constant: {Value}");
    }
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

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}Division:");
        writer.WriteLine($"{indent}  Left:");
        Left.Print(writer, indent + "    ");
        writer.WriteLine($"{indent}  Right:");
        Right.Print(writer, indent + "    ");
    }
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

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}Negation:");
        Operand.Print(writer, indent + "  ");
    }
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

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}PowerOp: ^{Exponent}");
        Base.Print(writer, indent + "  ");
    }
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

public class LinExpr : Expr
{
    public List<Expr> Terms { get; set; }
    public List<double> Weights { get; set; }
    public double ConstantTerm { get; set; }

    public LinExpr()
    {
        Terms = [];
        Weights = [];
    }
    
    public LinExpr(List<Expr> terms)
    {
        var constantSum = 0.0;
        var nonConstantTerms = new List<Expr>(terms.Count);
        var weights = new List<double>(terms.Count);

        foreach (var term in terms)
        {
            if (term is Constant c)
            {
                constantSum += c.Value;
            }
            // Extract weight from Negation
            else if (term is Negation neg)
            {
                // Recursively process the negated operand with weight -1
                ProcessTerm(neg.Operand, -1.0, ref constantSum, nonConstantTerms, weights);
            }
            // Merge nested LinExpr
            else if (term is LinExpr lin)
            {
                constantSum += lin.ConstantTerm;
                for (int i = 0; i < lin.Terms.Count; i++)
                {
                    nonConstantTerms.Add(lin.Terms[i]);
                    weights.Add(lin.Weights[i]);
                }
            }
            // Extract weight from Product of Constant * Expr
            else if (term is Product { Factors.Count: 2 } prod 
                     && prod.Factors[0] is Constant weight 
                     && prod.Factors[1] is not Constant)
            {
                ProcessTerm(prod.Factors[1], weight.Value, ref constantSum, nonConstantTerms, weights);
            }
            else if (term is Product { Factors.Count: 2 } prod2 
                     && prod2.Factors[1] is Constant weight2 
                     && prod2.Factors[0] is not Constant)
            {
                ProcessTerm(prod2.Factors[0], weight2.Value, ref constantSum, nonConstantTerms, weights);
            }
            else
            {
                nonConstantTerms.Add(term);
                weights.Add(1.0);
            }
        }

        Terms = nonConstantTerms;
        Weights = weights;
        ConstantTerm = constantSum;
    }

    private static void ProcessTerm(Expr term, double weight, ref double constantSum, List<Expr> nonConstantTerms, List<double> weights)
    {
        // Handle nested structures
        if (term is Constant c)
        {
            constantSum += weight * c.Value;
        }
        else if (term is Negation neg)
        {
            // Negate weight and continue
            ProcessTerm(neg.Operand, -weight, ref constantSum, nonConstantTerms, weights);
        }
        else if (term is LinExpr lin)
        {
            // Merge LinExpr: add all its terms with weights scaled by our weight
            constantSum += weight * lin.ConstantTerm;
            for (int i = 0; i < lin.Terms.Count; i++)
            {
                nonConstantTerms.Add(lin.Terms[i]);
                weights.Add(weight * lin.Weights[i]);
            }
        }
        else if (term is Product { Factors.Count: 2 } prod 
                 && prod.Factors[0] is Constant innerWeight 
                 && prod.Factors[1] is not Constant)
        {
            ProcessTerm(prod.Factors[1], weight * innerWeight.Value, ref constantSum, nonConstantTerms, weights);
        }
        else if (term is Product { Factors.Count: 2 } prod2 
                 && prod2.Factors[1] is Constant innerWeight2 
                 && prod2.Factors[0] is not Constant)
        {
            ProcessTerm(prod2.Factors[0], weight * innerWeight2.Value, ref constantSum, nonConstantTerms, weights);
        }
        else
        {
            nonConstantTerms.Add(term);
            weights.Add(weight);
        }
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var result = ConstantTerm;
        for (int i = 0; i < Terms.Count; i++)
            result += Weights[i] * Terms[i].Evaluate(x);
        return result;
    }

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        for (int i = 0; i < Terms.Count; i++)
            Terms[i].AccumulateGradient(x, grad, multiplier * Weights[i]);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        for (int i = 0; i < Terms.Count; i++)
            Terms[i].AccumulateHessian(x, hess, multiplier * Weights[i]);
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

    protected override Expr CloneCore()
    {
        var clone = new LinExpr([.. Terms]);
        clone.Weights = [.. Weights];
        clone.ConstantTerm = ConstantTerm;
        return clone;
    }

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}LinExpr: {Terms.Count} terms, constant={ConstantTerm}");
        for (int i = 0; i < Terms.Count; i++)
        {
            writer.WriteLine($"{indent}  [{i}] weight={Weights[i]}:");
            Terms[i].Print(writer, indent + "    ");
        }
    }
}

public class QuadExpr : Expr
{
    public List<Expr> LinearTerms { get; set; }
    public List<double> LinearWeights { get; set; }
    public List<Expr> QuadraticTerms1 { get; set; }
    public List<Expr> QuadraticTerms2 { get; set; }
    public List<double> QuadraticWeights { get; set; }
    public double ConstantTerm { get; set; }

    public QuadExpr()
    {
        LinearTerms = [];
        LinearWeights = [];
        QuadraticTerms1 = [];
        QuadraticTerms2 = [];
        QuadraticWeights = [];
        ConstantTerm = 0.0;
    }

    public QuadExpr(List<Expr> terms)
    {
        var constantSum = 0.0;
        var linearTerms = new List<Expr>();
        var linearWeights = new List<double>();
        var quadTerms1 = new List<Expr>();
        var quadTerms2 = new List<Expr>();
        var quadWeights = new List<double>();

        foreach (var term in terms)
        {
            if (term is Constant c)
            {
                constantSum += c.Value;
            }
            else if (term is Negation neg)
            {
                ProcessTerm(neg.Operand, -1.0, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
            }
            else if (term is LinExpr lin)
            {
                // Merge LinExpr - but check each term to see if it's actually quadratic
                constantSum += lin.ConstantTerm;
                for (int i = 0; i < lin.Terms.Count; i++)
                {
                    var linTerm = lin.Terms[i];
                    var linWeight = lin.Weights[i];
                    
                    // Check if this term is actually quadratic (shouldn't be in LinExpr!)
                    if (linTerm is Product prod)
                    {
                        // Process the product with the weight from LinExpr
                        ProcessProduct(prod, linWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                    }
                    else if (linTerm is PowerOp { Exponent: 2 } pow)
                    {
                        // x^2 term with weight
                        quadTerms1.Add(pow.Base);
                        quadTerms2.Add(pow.Base);
                        quadWeights.Add(linWeight);
                    }
                    else
                    {
                        // Actually linear
                        linearTerms.Add(linTerm);
                        linearWeights.Add(linWeight);
                    }
                }
            }
            else if (term is QuadExpr quad)
            {
                // Merge QuadExpr - must check linear terms for Products!
                constantSum += quad.ConstantTerm;
                for (int i = 0; i < quad.LinearTerms.Count; i++)
                {
                    var quadLinTerm = quad.LinearTerms[i];
                    var quadLinWeight = quad.LinearWeights[i];
                    
                    // Check if this linear term is actually quadratic (shouldn't be, but might be from old code)
                    if (quadLinTerm is Product prod)
                    {
                        ProcessProduct(prod, quadLinWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                    }
                    else if (quadLinTerm is PowerOp { Exponent: 2 } pow)
                    {
                        quadTerms1.Add(pow.Base);
                        quadTerms2.Add(pow.Base);
                        quadWeights.Add(quadLinWeight);
                    }
                    else
                    {
                        linearTerms.Add(quadLinTerm);
                        linearWeights.Add(quadLinWeight);
                    }
                }
                for (int i = 0; i < quad.QuadraticTerms1.Count; i++)
                {
                    quadTerms1.Add(quad.QuadraticTerms1[i]);
                    quadTerms2.Add(quad.QuadraticTerms2[i]);
                    quadWeights.Add(quad.QuadraticWeights[i]);
                }
            }
            else if (term is Product prod)
            {
                ProcessProduct(prod, 1.0, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
            }
            else if (term is PowerOp { Exponent: 2 } pow)
            {
                // x^2 → quadratic term
                quadTerms1.Add(pow.Base);
                quadTerms2.Add(pow.Base);
                quadWeights.Add(1.0);
            }
            else
            {
                // Fallback: add as linear term
                linearTerms.Add(term);
                linearWeights.Add(1.0);
            }
        }

        LinearTerms = linearTerms;
        LinearWeights = linearWeights;
        QuadraticTerms1 = quadTerms1;
        QuadraticTerms2 = quadTerms2;
        QuadraticWeights = quadWeights;
        ConstantTerm = constantSum;
    }

    private static void ProcessTerm(Expr term, double weight, ref double constantSum,
        List<Expr> linearTerms, List<double> linearWeights,
        List<Expr> quadTerms1, List<Expr> quadTerms2, List<double> quadWeights)
    {
        if (term is Constant c)
        {
            constantSum += weight * c.Value;
        }
        else if (term is Negation neg)
        {
            ProcessTerm(neg.Operand, -weight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
        }
        else if (term is LinExpr lin)
        {
            constantSum += weight * lin.ConstantTerm;
            for (int i = 0; i < lin.Terms.Count; i++)
            {
                // Don't just copy - recursively process each term in case it's a Product
                var linTerm = lin.Terms[i];
                var linWeight = lin.Weights[i];
                
                if (linTerm is Product prod)
                {
                    ProcessProduct(prod, weight * linWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                }
                else if (linTerm is PowerOp { Exponent: 2 } pow)
                {
                    quadTerms1.Add(pow.Base);
                    quadTerms2.Add(pow.Base);
                    quadWeights.Add(weight * linWeight);
                }
                else
                {
                    linearTerms.Add(linTerm);
                    linearWeights.Add(weight * linWeight);
                }
            }
        }
        else if (term is QuadExpr quad)
        {
            constantSum += weight * quad.ConstantTerm;
            for (int i = 0; i < quad.LinearTerms.Count; i++)
            {
                // Don't just copy - recursively process each linear term in case it's a Product
                var quadLinTerm = quad.LinearTerms[i];
                var quadLinWeight = quad.LinearWeights[i];
                
                if (quadLinTerm is Product prod)
                {
                    ProcessProduct(prod, weight * quadLinWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                }
                else if (quadLinTerm is PowerOp { Exponent: 2 } pow)
                {
                    quadTerms1.Add(pow.Base);
                    quadTerms2.Add(pow.Base);
                    quadWeights.Add(weight * quadLinWeight);
                }
                else
                {
                    linearTerms.Add(quadLinTerm);
                    linearWeights.Add(weight * quadLinWeight);
                }
            }
            for (int i = 0; i < quad.QuadraticTerms1.Count; i++)
            {
                quadTerms1.Add(quad.QuadraticTerms1[i]);
                quadTerms2.Add(quad.QuadraticTerms2[i]);
                quadWeights.Add(weight * quad.QuadraticWeights[i]);
            }
        }
        else if (term is Product prod)
        {
            ProcessProduct(prod, weight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
        }
        else if (term is PowerOp { Exponent: 2 } pow)
        {
            quadTerms1.Add(pow.Base);
            quadTerms2.Add(pow.Base);
            quadWeights.Add(weight);
        }
        else
        {
            linearTerms.Add(term);
            linearWeights.Add(weight);
        }
    }

    private static void ProcessProduct(Product prod, double weight, ref double constantSum,
        List<Expr> linearTerms, List<double> linearWeights,
        List<Expr> quadTerms1, List<Expr> quadTerms2, List<double> quadWeights)
    {
        // Extract all constants from the product
        var productWeight = weight;
        var nonConstants = new List<Expr>();
        
        foreach (var factor in prod.Factors)
        {
            if (factor is Constant c)
                productWeight *= c.Value;
            else
                nonConstants.Add(factor);
        }

        // Handle based on number of non-constant factors
        if (nonConstants.Count == 0)
        {
            // All constants
            constantSum += productWeight;
        }
        else if (nonConstants.Count == 1)
        {
            // Linear term: c * expr
            var expr = nonConstants[0];
            if (expr is PowerOp { Exponent: 2 } pow)
            {
                // c * x^2
                quadTerms1.Add(pow.Base);
                quadTerms2.Add(pow.Base);
                quadWeights.Add(productWeight);
            }
            else
            {
                ProcessTerm(expr, productWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
            }
        }
        else if (nonConstants.Count == 2)
        {
            // Bilinear or quadratic term: c * expr1 * expr2
            var expr1 = nonConstants[0];
            var expr2 = nonConstants[1];

            // Check for x^2 represented as x*x
            if (expr1 is Variable v1 && expr2 is Variable v2 && v1.Index == v2.Index)
            {
                quadTerms1.Add(expr1);
                quadTerms2.Add(expr1);
                quadWeights.Add(productWeight);
            }
            // Expand LinExpr * LinExpr
            else if (expr1 is LinExpr lin1 && expr2 is LinExpr lin2)
            {
                // Expand: (Σ a_i * x_i + c1) * (Σ b_j * y_j + c2)
                // = Σ_i Σ_j (a_i * b_j * x_i * y_j) + Σ_i (a_i * c2 * x_i) + Σ_j (b_j * c1 * y_j) + c1 * c2
                
                // Constant * Constant
                constantSum += productWeight * lin1.ConstantTerm * lin2.ConstantTerm;
                
                // Linear terms from first LinExpr * constant from second
                for (int i = 0; i < lin1.Terms.Count; i++)
                {
                    // Don't just add the term - process it in case it's a Product
                    ProcessTerm(lin1.Terms[i], productWeight * lin1.Weights[i] * lin2.ConstantTerm, 
                        ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                }
                
                // Linear terms from second LinExpr * constant from first
                for (int j = 0; j < lin2.Terms.Count; j++)
                {
                    // Don't just add the term - process it in case it's a Product
                    ProcessTerm(lin2.Terms[j], productWeight * lin2.Weights[j] * lin1.ConstantTerm, 
                        ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                }
                
                // Quadratic cross terms
                for (int i = 0; i < lin1.Terms.Count; i++)
                {
                    for (int j = 0; j < lin2.Terms.Count; j++)
                    {
                        quadTerms1.Add(lin1.Terms[i]);
                        quadTerms2.Add(lin2.Terms[j]);
                        quadWeights.Add(productWeight * lin1.Weights[i] * lin2.Weights[j]);
                    }
                }
            }
            // Expand LinExpr * other
            else if (expr1 is LinExpr lin)
            {
                // (Σ a_i * x_i + c) * y = Σ (a_i * x_i * y) + c * y
                // Add c * y as a linear or quadratic term depending on y
                if (lin.ConstantTerm != 0.0)
                {
                    ProcessTerm(expr2, productWeight * lin.ConstantTerm, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                }
                
                for (int i = 0; i < lin.Terms.Count; i++)
                {
                    // Create bilinear term, but the term from LinExpr might itself be a Product
                    // so we need to check
                    var linTerm = lin.Terms[i];
                    var linWeight = lin.Weights[i];
                    
                    if (linTerm is Product nestedProd)
                    {
                        // This is a Product in a LinExpr - we need to expand it with expr2
                        // First process the nested product to get its components
                        var tempLinTerms = new List<Expr>();
                        var tempLinWeights = new List<double>();
                        var tempQuadTerms1 = new List<Expr>();
                        var tempQuadTerms2 = new List<Expr>();
                        var tempQuadWeights = new List<double>();
                        var tempConst = 0.0;
                        
                        ProcessProduct(nestedProd, linWeight, ref tempConst, tempLinTerms, tempLinWeights, tempQuadTerms1, tempQuadTerms2, tempQuadWeights);
                        
                        // Now multiply each result by expr2
                        for (int k = 0; k < tempLinTerms.Count; k++)
                        {
                            quadTerms1.Add(tempLinTerms[k]);
                            quadTerms2.Add(expr2);
                            quadWeights.Add(productWeight * tempLinWeights[k]);
                        }
                        // Quadratic terms from the nested product can't be multiplied by expr2
                        // (that would be cubic), so just add them scaled
                        for (int k = 0; k < tempQuadTerms1.Count; k++)
                        {
                            // This creates a higher-order term - can't represent in QuadExpr
                            // Add back as a Product
                            var higherOrder = new Product([new Constant(tempQuadWeights[k]), tempQuadTerms1[k], tempQuadTerms2[k], expr2]);
                            linearTerms.Add(higherOrder);
                            linearWeights.Add(productWeight);
                        }
                    }
                    else
                    {
                        // Normal case: linear term * expr2
                        quadTerms1.Add(linTerm);
                        quadTerms2.Add(expr2);
                        quadWeights.Add(productWeight * linWeight);
                    }
                }
            }
            // Expand other * LinExpr
            else if (expr2 is LinExpr linRight)
            {
                if (linRight.ConstantTerm != 0.0)
                {
                    ProcessTerm(expr1, productWeight * linRight.ConstantTerm, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                }
                
                for (int i = 0; i < linRight.Terms.Count; i++)
                {
                    // Same as above - check if the term is a Product
                    var linTerm = linRight.Terms[i];
                    var linWeight = linRight.Weights[i];
                    
                    if (linTerm is Product nestedProd)
                    {
                        // Process the nested product and multiply by expr1
                        var tempLinTerms = new List<Expr>();
                        var tempLinWeights = new List<double>();
                        var tempQuadTerms1 = new List<Expr>();
                        var tempQuadTerms2 = new List<Expr>();
                        var tempQuadWeights = new List<double>();
                        var tempConst = 0.0;
                        
                        ProcessProduct(nestedProd, linWeight, ref tempConst, tempLinTerms, tempLinWeights, tempQuadTerms1, tempQuadTerms2, tempQuadWeights);
                        
                        for (int k = 0; k < tempLinTerms.Count; k++)
                        {
                            quadTerms1.Add(expr1);
                            quadTerms2.Add(tempLinTerms[k]);
                            quadWeights.Add(productWeight * tempLinWeights[k]);
                        }
                        // Higher-order terms
                        for (int k = 0; k < tempQuadTerms1.Count; k++)
                        {
                            var higherOrder = new Product([new Constant(tempQuadWeights[k]), expr1, tempQuadTerms1[k], tempQuadTerms2[k]]);
                            linearTerms.Add(higherOrder);
                            linearWeights.Add(productWeight);
                        }
                    }
                    else
                    {
                        quadTerms1.Add(expr1);
                        quadTerms2.Add(linTerm);
                        quadWeights.Add(productWeight * linWeight);
                    }
                }
            }
            else
            {
                // General bilinear: x * y
                quadTerms1.Add(expr1);
                quadTerms2.Add(expr2);
                quadWeights.Add(productWeight);
            }
        }
        else
        {
            // 3+ non-constant factors: this is higher than quadratic
            // Reconstruct the product and add as linear term (will fail IsAtMostQuadratic check later)
            var reconstructed = nonConstants.Count == prod.Factors.Count 
                ? (Expr)prod 
                : new Product([new Constant(productWeight / weight), .. nonConstants]);
            linearTerms.Add(reconstructed);
            linearWeights.Add(weight);
        }
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var result = ConstantTerm;
        
        // Linear terms
        for (int i = 0; i < LinearTerms.Count; i++)
            result += LinearWeights[i] * LinearTerms[i].Evaluate(x);
        
        // Quadratic terms
        for (int i = 0; i < QuadraticTerms1.Count; i++)
            result += QuadraticWeights[i] * QuadraticTerms1[i].Evaluate(x) * QuadraticTerms2[i].Evaluate(x);
        
        return result;
    }

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        // Linear terms
        for (int i = 0; i < LinearTerms.Count; i++)
            LinearTerms[i].AccumulateGradient(x, grad, multiplier * LinearWeights[i]);
        
        // Quadratic terms: d/dx_i (w * f * g) = w * (f' * g + f * g')
        for (int i = 0; i < QuadraticTerms1.Count; i++)
        {
            var f = QuadraticTerms1[i].Evaluate(x);
            var g = QuadraticTerms2[i].Evaluate(x);
            var w = multiplier * QuadraticWeights[i];
            
            QuadraticTerms1[i].AccumulateGradient(x, grad, w * g);
            QuadraticTerms2[i].AccumulateGradient(x, grad, w * f);
        }
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // Linear terms contribute nothing to Hessian
        
        // Quadratic terms: ∂²/∂x_i∂x_j (w * f * g)
        // For simple variable products (most common case): x_i * x_j → coefficient w
        for (int i = 0; i < QuadraticTerms1.Count; i++)
        {
            var w = multiplier * QuadraticWeights[i];
            
            // Simplified for Variable * Variable case (most common for quadratic expressions)
            if (QuadraticTerms1[i] is Variable v1 && QuadraticTerms2[i] is Variable v2)
            {
                int idx1 = v1.Index;
                int idx2 = v2.Index;
                
                if (idx1 == idx2)
                {
                    // Diagonal: x^2 → d²/dx² = 2
                    hess.Add(idx1, idx1, 2.0 * w);
                }
                else
                {
                    // Off-diagonal: x*y → d²/dxdy = 1 (coefficient stored once, used for both i,j and j,i)
                    hess.Add(idx1, idx2, w);
                }
            }
            else
            {
                // For complex expressions, delegate to their Hessian computation
                // This handles cases like (expr1 * expr2) where expr1/expr2 aren't simple variables
                QuadraticTerms1[i].AccumulateHessian(x, hess, w * QuadraticTerms2[i].Evaluate(x));
                QuadraticTerms2[i].AccumulateHessian(x, hess, w * QuadraticTerms1[i].Evaluate(x));
            }
        }
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        foreach (var term in LinearTerms)
            term.CollectVariables(variables);
        foreach (var term in QuadraticTerms1)
            term.CollectVariables(variables);
        foreach (var term in QuadraticTerms2)
            term.CollectVariables(variables);
    }

    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        foreach (var term in LinearTerms)
            term.CollectHessianSparsity(entries);
        
        // Quadratic terms contribute to Hessian sparsity
        for (int i = 0; i < QuadraticTerms1.Count; i++)
        {
            if (QuadraticTerms1[i] is Variable v1 && QuadraticTerms2[i] is Variable v2)
            {
                int idx1 = v1.Index;
                int idx2 = v2.Index;
                int row = Math.Min(idx1, idx2);
                int col = Math.Max(idx1, idx2);
                entries.Add((row, col));
            }
            else
            {
                QuadraticTerms1[i].CollectHessianSparsity(entries);
                QuadraticTerms2[i].CollectHessianSparsity(entries);
            }
        }
    }

    protected override bool IsConstantWrtXCore() => 
        LinearTerms.All(t => t.IsConstantWrtX()) && 
        QuadraticTerms1.All(t => t.IsConstantWrtX()) && 
        QuadraticTerms2.All(t => t.IsConstantWrtX());
    
    protected override bool IsLinearCore() => QuadraticTerms1.Count == 0 && LinearTerms.All(t => t.IsLinear());
    protected override bool IsAtMostQuadraticCore() => 
        LinearTerms.All(t => t.IsAtMostQuadratic()) && 
        QuadraticTerms1.All(t => t.IsLinear()) && 
        QuadraticTerms2.All(t => t.IsLinear());

    protected override Expr CloneCore()
    {
        var clone = new QuadExpr();
        clone.LinearTerms = [.. LinearTerms];
        clone.LinearWeights = [.. LinearWeights];
        clone.QuadraticTerms1 = [.. QuadraticTerms1];
        clone.QuadraticTerms2 = [.. QuadraticTerms2];
        clone.QuadraticWeights = [.. QuadraticWeights];
        clone.ConstantTerm = ConstantTerm;
        return clone;
    }

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}QuadExpr: {LinearTerms.Count} linear, {QuadraticTerms1.Count} quadratic, constant={ConstantTerm}");
        
        for (int i = 0; i < LinearTerms.Count; i++)
        {
            writer.WriteLine($"{indent}  Linear [{i}] weight={LinearWeights[i]}:");
            LinearTerms[i].Print(writer, indent + "    ");
        }
        
        for (int i = 0; i < QuadraticTerms1.Count; i++)
        {
            writer.WriteLine($"{indent}  Quadratic [{i}] weight={QuadraticWeights[i]}:");
            writer.WriteLine($"{indent}    Term1:");
            QuadraticTerms1[i].Print(writer, indent + "      ");
            writer.WriteLine($"{indent}    Term2:");
            QuadraticTerms2[i].Print(writer, indent + "      ");
        }
    }
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

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}Product: {Factors.Count} factors");
        for (int i = 0; i < Factors.Count; i++)
        {
            writer.WriteLine($"{indent}  [{i}]:");
            Factors[i].Print(writer, indent + "    ");
        }
    }
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
