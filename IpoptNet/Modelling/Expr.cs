using System.Runtime.CompilerServices;
using System.Buffers;

namespace IpoptNet.Modelling;

public abstract class Expr
{
    protected Expr? _replacement;
    internal HashSet<Variable>? _cachedVariables;
    internal int[]? _sortedVarIndices;

    /// <summary>
    /// Gets the actual expression, following any replacements. For testing purposes.
    /// </summary>
    public Expr GetActual() => _replacement ?? this;

    public double Evaluate(ReadOnlySpan<double> x) => _replacement?.Evaluate(x) ?? EvaluateCore(x);

    public void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad)
    {
        var expr = _replacement ?? this;

        // Allocate compact buffer and compute compact gradient
        var compactGrad = ArrayPool<double>.Shared.Rent(expr._cachedVariables!.Count);
        Array.Clear(compactGrad, 0, expr._cachedVariables!.Count);
        expr.AccumulateGradientCompactCore(x, compactGrad, 1.0, expr._sortedVarIndices!);

        // Expand compact gradient to full-sized array
        for (int i = 0; i < expr._sortedVarIndices!.Length; i++)
            grad[expr._sortedVarIndices[i]] += compactGrad[i];

        ArrayPool<double>.Shared.Return(compactGrad);
    }

    public void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier) =>
        (_replacement ?? this).AccumulateHessianCore(x, hess, multiplier);
    public void CollectVariables(HashSet<Variable> variables) =>
        (_replacement ?? this).CollectVariablesCore(variables);
    public void CollectHessianSparsity(HashSet<(int row, int col)> entries) =>
        (_replacement ?? this).CollectHessianSparsityCore(entries);

    internal void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices) =>
        (_replacement ?? this).AccumulateGradientCompactCore(x, compactGrad, multiplier, sortedVarIndices);

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
    protected abstract void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices);
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
        a = a.GetActual();
        b = b.GetActual();

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
        a = a.GetActual();
        b = b.GetActual();

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
        a = a.GetActual();
        b = b.GetActual();

        // If multiplying by a constant, use the optimized scalar multiplication
        if (b is Constant c)
            return a * c.Value;
        if (a is Constant c2)
            return c2.Value * b;

        // If left operand is already a Product, extend it with the right operand
        if (a is Product prodA)
        {
            return new Product([.. prodA.Factors, b])
            {
                Factor = prodA.Factor
            };
        }
        // If right operand is a Product, prepend the left operand to it
        else if (b is Product prodB)
        {
            return new Product([a, .. prodB.Factors])
            {
                Factor = prodB.Factor
            };
        }
        else if (ReferenceEquals(a, b))
        {
            return new PowerOp(a, 2);
        }
        // Always use Product for multiplication to avoid building deep trees
        else
        {
            return new Product([a, b]);
        }
    }
    public static Expr operator /(Expr a, Expr b)
    {
        a = a.GetActual();
        b = b.GetActual();

        // If dividing by a constant, use the optimized scalar division
        if (b is Constant c)
            return a / c.Value;
        return new Division(a, b);
    }
    public static Expr operator -(Expr a) => new Negation(a);

    public static Expr operator +(Expr a, double b)
    {
        a = a.GetActual();

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
        b = b.GetActual();

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
        a = a.GetActual();

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
        b = b.GetActual();

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
        a = a.GetActual();

        if (a is LinExpr linA)
        {
            return new LinExpr()
            {
                Terms = [.. linA.Terms],
                Weights = linA.Weights.Select(w => w * b).ToList(),
                ConstantTerm = linA.ConstantTerm * b
            };
        }
        if (a is QuadExpr quadA)
        {
            return new QuadExpr()
            {
                LinearTerms = [.. quadA.LinearTerms],
                LinearWeights = quadA.LinearWeights.Select(w => w * b).ToList(),
                QuadraticTerms1 = [.. quadA.QuadraticTerms1],
                QuadraticTerms2 = [.. quadA.QuadraticTerms2],
                QuadraticWeights = quadA.QuadraticWeights.Select(w => w * b).ToList(),
                ConstantTerm = quadA.ConstantTerm * b
            };
        }
        if (a is Product prodA)
        {
            return new Product([.. prodA.Factors])
            {
                Factor = prodA.Factor * b
            };
        }
        return new Product([a, new Constant(b)]);
    }

    public static Expr operator *(double a, Expr b)
    {
        b = b.GetActual();

        if (b is LinExpr linB)
        {
            return new LinExpr()
            {
                Terms = [.. linB.Terms],
                Weights = linB.Weights.Select(w => w * a).ToList(),
                ConstantTerm = linB.ConstantTerm * a
            };
        }
        if (b is QuadExpr quadB)
        {
            return new QuadExpr()
            {
                LinearTerms = [.. quadB.LinearTerms],
                LinearWeights = quadB.LinearWeights.Select(w => w * a).ToList(),
                QuadraticTerms1 = [.. quadB.QuadraticTerms1],
                QuadraticTerms2 = [.. quadB.QuadraticTerms2],
                QuadraticWeights = quadB.QuadraticWeights.Select(w => w * a).ToList(),
                ConstantTerm = quadB.ConstantTerm * a
            };
        }
        if (b is Product prodB)
        {
            return new Product([.. prodB.Factors])
            {
                Factor = prodB.Factor * a
            };
        }
        return new Product([new Constant(a), b]);
    }
    public static Expr operator /(Expr a, double b)
    {
        a = a.GetActual();

        if (a is LinExpr linA)
        {
            return new LinExpr()
            {
                Terms = [.. linA.Terms],
                Weights = linA.Weights.Select(w => w / b).ToList(),
                ConstantTerm = linA.ConstantTerm / b
            };
        }
        if (a is QuadExpr quadA)
        {
            return new QuadExpr()
            {
                LinearTerms = [.. quadA.LinearTerms],
                LinearWeights = quadA.LinearWeights.Select(w => w / b).ToList(),
                QuadraticTerms1 = [.. quadA.QuadraticTerms1],
                QuadraticTerms2 = [.. quadA.QuadraticTerms2],
                QuadraticWeights = quadA.QuadraticWeights.Select(w => w / b).ToList(),
                ConstantTerm = quadA.ConstantTerm / b
            };
        }
        return a * (1.0 / b);
    }
    public static Expr operator /(double a, Expr b) => new Division(new Constant(a), b);

    // C# 14 compound assignment operators - modify expression in-place for efficiency
    public void operator +=(Expr other)
    {
        other = other.GetActual();

        // Check if we've been replaced with a QuadExpr - use efficient AddTerm
        if (_replacement is QuadExpr quad)
        {
            quad.AddTerm(other, 1.0);
        }
        // Check if we've been replaced with a LinExpr - use efficient AddTerm
        else if (_replacement is LinExpr lin)
        {
            lin.AddTerm(other, 1.0);
        }
        // Check if this is directly a QuadExpr (no replacement) - use efficient AddTerm
        else if (_replacement is null && this is QuadExpr thisQuad)
        {
            thisQuad.AddTerm(other, 1.0);
        }
        // Check if this is directly a LinExpr (no replacement) - use efficient AddTerm
        else if (_replacement is null && this is LinExpr thisLin)
        {
            thisLin.AddTerm(other, 1.0);
        }
        // Check if this is a zero constant with no replacement yet
        else if (_replacement is null && this is Constant { Value: 0 })
        {
            ReplaceWith(other);
        }
        // Otherwise create appropriate expression type
        else
        {
            var current = Clone();
            // Use QuadExpr if either operand is quadratic
            if ((!current.IsLinear() && current.IsAtMostQuadratic()) ||
                (!other.IsLinear() && other.IsAtMostQuadratic()) ||
                current is QuadExpr || other is QuadExpr)
            {
                ReplaceWith(new QuadExpr([current, other]));
            }
            else
            {
                ReplaceWith(new LinExpr([current, other]));
            }
        }
    }

    public void operator -=(Expr other)
    {
        other = other.GetActual();

        // Check if we've been replaced with a QuadExpr - use efficient AddTerm
        if (_replacement is QuadExpr quad)
        {
            quad.AddTerm(other, -1.0);
        }
        // Check if we've been replaced with a LinExpr - use efficient AddTerm
        else if (_replacement is LinExpr lin)
        {
            lin.AddTerm(other, -1.0);
        }
        // Check if this is directly a QuadExpr (no replacement) - use efficient AddTerm
        else if (_replacement is null && this is QuadExpr thisQuad)
        {
            thisQuad.AddTerm(other, -1.0);
        }
        // Check if this is directly a LinExpr (no replacement) - use efficient AddTerm
        else if (_replacement is null && this is LinExpr thisLin)
        {
            thisLin.AddTerm(other, -1.0);
        }
        // Check if this is a zero constant with no replacement yet
        else if (_replacement is null && this is Constant { Value: 0 })
        {
            ReplaceWith(-other);
        }
        // Otherwise create appropriate expression type
        else
        {
            var current = Clone();
            // Use QuadExpr if either operand is quadratic
            if ((!current.IsLinear() && current.IsAtMostQuadratic()) ||
                (!other.IsLinear() && other.IsAtMostQuadratic()) ||
                current is QuadExpr || other is QuadExpr)
            {
                ReplaceWith(new QuadExpr([current, -other]));
            }
            else
            {
                ReplaceWith(new LinExpr([current, -other]));
            }
        }
    }

    public void operator *=(Expr other)
    {
        other = other.GetActual();

        // Special handling for LinExpr and QuadExpr multiplying by constant
        if (other is Constant c)
        {
            if (_replacement is LinExpr replacementLin)
            {
                for (int i = 0; i < replacementLin.Weights.Count; i++)
                    replacementLin.Weights[i] *= c.Value;
                replacementLin.ConstantTerm *= c.Value;
                return;
            }
            if (_replacement is QuadExpr replacementQuad)
            {
                for (int i = 0; i < replacementQuad.LinearWeights.Count; i++)
                    replacementQuad.LinearWeights[i] *= c.Value;
                for (int i = 0; i < replacementQuad.QuadraticWeights.Count; i++)
                    replacementQuad.QuadraticWeights[i] *= c.Value;
                replacementQuad.ConstantTerm *= c.Value;
                return;
            }
            if (_replacement is null && this is LinExpr thisLin)
            {
                for (int i = 0; i < thisLin.Weights.Count; i++)
                    thisLin.Weights[i] *= c.Value;
                thisLin.ConstantTerm *= c.Value;
                return;
            }
            if (_replacement is null && this is QuadExpr thisQuad)
            {
                for (int i = 0; i < thisQuad.LinearWeights.Count; i++)
                    thisQuad.LinearWeights[i] *= c.Value;
                for (int i = 0; i < thisQuad.QuadraticWeights.Count; i++)
                    thisQuad.QuadraticWeights[i] *= c.Value;
                thisQuad.ConstantTerm *= c.Value;
                return;
            }
        }

        // Check if we've been replaced with a Product
        if (_replacement is Product replacementProduct)
        {
            replacementProduct.Factors.Add(other);
        }
        // Check if this is directly a Product (no replacement)
        else if (_replacement is null && this is Product thisProduct)
        {
            thisProduct.Factors.Add(other);
        }
        // Check if this is a one constant with no replacement yet
        else if (_replacement is null && this is Constant { Value: 1 })
        {
            ReplaceWith(other);
        }
        // Otherwise create a new Product with current value and new factor
        else
        {
            ReplaceWith(new Product([Clone(), other]));
        }
    }

    public void operator /=(Expr other)
    {
        other = other.GetActual();

        // Special handling for LinExpr and QuadExpr dividing by constant
        if (other is Constant c)
        {
            if (_replacement is LinExpr replacementLin)
            {
                for (int i = 0; i < replacementLin.Weights.Count; i++)
                    replacementLin.Weights[i] /= c.Value;
                replacementLin.ConstantTerm /= c.Value;
                return;
            }
            if (_replacement is QuadExpr replacementQuad)
            {
                for (int i = 0; i < replacementQuad.LinearWeights.Count; i++)
                    replacementQuad.LinearWeights[i] /= c.Value;
                for (int i = 0; i < replacementQuad.QuadraticWeights.Count; i++)
                    replacementQuad.QuadraticWeights[i] /= c.Value;
                replacementQuad.ConstantTerm /= c.Value;
                return;
            }
            if (_replacement is null && this is LinExpr thisLin)
            {
                for (int i = 0; i < thisLin.Weights.Count; i++)
                    thisLin.Weights[i] /= c.Value;
                thisLin.ConstantTerm /= c.Value;
                return;
            }
            if (_replacement is null && this is QuadExpr thisQuad)
            {
                for (int i = 0; i < thisQuad.LinearWeights.Count; i++)
                    thisQuad.LinearWeights[i] /= c.Value;
                for (int i = 0; i < thisQuad.QuadraticWeights.Count; i++)
                    thisQuad.QuadraticWeights[i] /= c.Value;
                thisQuad.ConstantTerm /= c.Value;
                return;
            }
        }

        // Check if we've been replaced with a Product (add reciprocal)
        if (_replacement is Product replacementProduct)
        {
            replacementProduct.Factors.Add(new Division(1, other));
        }
        // Check if this is directly a Product (no replacement)
        else if (_replacement is null && this is Product thisProduct)
        {
            thisProduct.Factors.Add(new Division(1, other));
        }
        // Otherwise create a division
        else
        {
            ReplaceWith(new Division(Clone(), other));
        }
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

    /// <summary>
    /// Caches variables for this expression and all children to optimize repeated Hessian evaluations.
    /// Called once during model finalization before optimization.
    /// </summary>
    internal void Prepare()
    {
        if (_replacement is not null)
        {
            _replacement.Prepare();
            return;
        }

        if (_cachedVariables is not null)
            return;

        _cachedVariables = new HashSet<Variable>();
        CollectVariablesCore(_cachedVariables);

        // Build sorted variable indices
        _sortedVarIndices = new int[_cachedVariables.Count];
        {
            var i = 0;
            foreach (var v in _cachedVariables)
                _sortedVarIndices[i++] = v.Index;
        }
        Array.Sort(_sortedVarIndices);

        // Recursively cache for children
        PrepareChildren();
    }

    /// <summary>
    /// Clears cached variables to free memory after optimization completes.
    /// </summary>
    internal void Clear()
    {
        if (_replacement is not null)
            _replacement.Clear();

        _cachedVariables = null;
        _sortedVarIndices = null;
        ClearChildren();
    }

    protected virtual void PrepareChildren() { }

    protected virtual void ClearChildren() { }

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

    public override string ToString()
    {
        if (_replacement is not null)
            return _replacement.ToString();
        return ToStringCore();
    }

    protected virtual string ToStringCore() => GetType().Name;

    /// <summary>
    /// Returns true if this expression contains only variables and constants (no complex operations).
    /// </summary>
    internal virtual bool IsSimpleForPrinting() => false;
}
