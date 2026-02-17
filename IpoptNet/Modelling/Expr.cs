using System.Runtime.CompilerServices;
using System.Buffers;

namespace IpoptNet.Modelling;

public sealed class Expr
{
    internal ExprNode _node;

    internal Expr(ExprNode node)
    {
        _node = node;
    }

    public double Evaluate(ReadOnlySpan<double> x) => _node.Evaluate(x);

    public void AccumulateGradient(ReadOnlySpan<double> x, Span<double> grad)
    {
        // Allocate compact buffer and compute compact gradient
        var compactGrad = ArrayPool<double>.Shared.Rent(_node._cachedVariables!.Count);
        Array.Clear(compactGrad, 0, _node._cachedVariables!.Count);
        _node.AccumulateGradientCompact(x, compactGrad, 1.0, _node._sortedVarIndices!);

        // Expand compact gradient to full-sized array
        for (int i = 0; i < _node._sortedVarIndices!.Length; i++)
            grad[_node._sortedVarIndices[i]] += compactGrad[i];

        ArrayPool<double>.Shared.Return(compactGrad);
    }

    public void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier) =>
        _node.AccumulateHessian(x, hess, multiplier);
    public void CollectVariables(HashSet<Variable> variables) =>
        _node.CollectVariables(variables);
    public void CollectHessianSparsity(HashSet<(int row, int col)> entries) =>
        _node.CollectHessianSparsity(entries);

    /// <summary>
    /// Returns true if this expression contains no variables (is a constant value).
    /// </summary>
    public bool IsConstantWrtX() => _node.IsConstantWrtX();

    /// <summary>
    /// Returns true if this expression is linear (constant gradient, zero Hessian).
    /// Examples: 2*x + 3*y - 5, x, Constant
    /// </summary>
    public bool IsLinear() => _node.IsLinear();

    /// <summary>
    /// Returns true if this expression is at most quadratic (constant Hessian).
    /// Examples: x*y, x^2, 2*x + 3*y - 5
    /// </summary>
    public bool IsAtMostQuadratic() => _node.IsAtMostQuadratic();

    public override bool Equals(object? obj) => ReferenceEquals(this, obj);
    public override int GetHashCode() => RuntimeHelpers.GetHashCode(this);

    public static implicit operator Expr(int value) => new Expr(new ConstantNode(value));
    public static implicit operator Expr(double value) => new Expr(new ConstantNode(value));
    public static implicit operator Expr(Variable v) => v._expr;

    public static Expr operator +(Expr a, Expr b)
    {
        var na = a._node;
        var nb = b._node;

        // If either operand is QuadExpr, result should be QuadExpr
        // BUT only if the other operand is also at most quadratic
        if (na is QuadExprNode && nb.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([na, nb]));
        }
        if (nb is QuadExprNode && na.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([na, nb]));
        }

        // Auto-create QuadExpr if BOTH operands are at most quadratic
        // and at least one is actually quadratic (non-linear)
        bool aIsQuadratic = !na.IsLinear() && na.IsAtMostQuadratic();
        bool bIsQuadratic = !nb.IsLinear() && nb.IsAtMostQuadratic();

        if ((aIsQuadratic || bIsQuadratic) && na.IsAtMostQuadratic() && nb.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([na, nb]));
        }

        // Always create a new LinExpr to ensure proper merging
        return new Expr(new LinExprNode([na, nb]));
    }

    public static Expr operator -(Expr a, Expr b)
    {
        var na = a._node;
        var nb = b._node;

        // If either operand is QuadExpr, result should be QuadExpr
        // BUT only if the other operand is also at most quadratic
        if (na is QuadExprNode && nb.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([na, new NegationNode(nb)]));
        }
        if (nb is QuadExprNode && na.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([na, new NegationNode(nb)]));
        }

        // Auto-create QuadExpr if BOTH operands are at most quadratic
        bool aIsQuadratic = !na.IsLinear() && na.IsAtMostQuadratic();
        bool bIsQuadratic = !nb.IsLinear() && nb.IsAtMostQuadratic();

        if ((aIsQuadratic || bIsQuadratic) && na.IsAtMostQuadratic() && nb.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([na, new NegationNode(nb)]));
        }

        // Always create a new LinExpr to ensure proper merging
        return new Expr(new LinExprNode([na, new NegationNode(nb)]));
    }
    public static Expr operator *(Expr a, Expr b)
    {
        var na = a._node;
        var nb = b._node;

        // If multiplying by a constant, use the optimized scalar multiplication
        if (nb is ConstantNode c)
            return a * c.Value;
        if (na is ConstantNode c2)
            return c2.Value * b;

        // If left operand is already a Product, extend it with the right operand
        if (na is ProductNode prodA)
        {
            return new Expr(new ProductNode([.. prodA.Factors, nb])
            {
                Factor = prodA.Factor
            });
        }
        // If right operand is a Product, prepend the left operand to it
        else if (nb is ProductNode prodB)
        {
            return new Expr(new ProductNode([na, .. prodB.Factors])
            {
                Factor = prodB.Factor
            });
        }
        else if (ReferenceEquals(na, nb))
        {
            return new Expr(new PowerOpNode(na, 2));
        }
        // Always use Product for multiplication to avoid building deep trees
        else
        {
            return new Expr(new ProductNode([na, nb]));
        }
    }
    public static Expr operator /(Expr a, Expr b)
    {
        var na = a._node;
        var nb = b._node;

        // If dividing by a constant, use the optimized scalar division
        if (nb is ConstantNode c)
            return a / c.Value;
        return new Expr(new DivisionNode(na, nb));
    }
    public static Expr operator -(Expr a) => new Expr(new NegationNode(a._node));

    public static Expr operator +(Expr a, double b)
    {
        var na = a._node;

        // Preserve QuadExpr if present
        if (na is QuadExprNode)
        {
            return new Expr(new QuadExprNode([na, new ConstantNode(b)]));
        }
        // Auto-create QuadExpr for quadratic expressions
        if (!na.IsLinear() && na.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([na, new ConstantNode(b)]));
        }
        return new Expr(new LinExprNode([na, new ConstantNode(b)]));
    }

    public static Expr operator +(double a, Expr b)
    {
        var nb = b._node;

        // Preserve QuadExpr if present
        if (nb is QuadExprNode)
        {
            return new Expr(new QuadExprNode([new ConstantNode(a), nb]));
        }
        // Auto-create QuadExpr for quadratic expressions
        if (!nb.IsLinear() && nb.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([new ConstantNode(a), nb]));
        }
        return new Expr(new LinExprNode([new ConstantNode(a), nb]));
    }
    public static Expr operator -(Expr a, double b)
    {
        var na = a._node;

        // Preserve QuadExpr if present
        if (na is QuadExprNode)
        {
            return new Expr(new QuadExprNode([na, new ConstantNode(-b)]));
        }
        // Auto-create QuadExpr for quadratic expressions
        if (!na.IsLinear() && na.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([na, new ConstantNode(-b)]));
        }
        return new Expr(new LinExprNode([na, new ConstantNode(-b)]));
    }

    public static Expr operator -(double a, Expr b)
    {
        var nb = b._node;

        // Preserve QuadExpr if present
        if (nb is QuadExprNode)
        {
            return new Expr(new QuadExprNode([new ConstantNode(a), new NegationNode(nb)]));
        }
        // Auto-create QuadExpr for quadratic expressions
        if (!nb.IsLinear() && nb.IsAtMostQuadratic())
        {
            return new Expr(new QuadExprNode([new ConstantNode(a), new NegationNode(nb)]));
        }
        return new Expr(new LinExprNode([new ConstantNode(a), new NegationNode(nb)]));
    }
    public static Expr operator *(Expr a, double b)
    {
        var na = a._node;

        if (na is LinExprNode linA)
        {
            return new Expr(new LinExprNode()
            {
                Terms = [.. linA.Terms],
                Weights = linA.Weights.Select(w => w * b).ToList(),
                ConstantTerm = linA.ConstantTerm * b
            });
        }
        if (na is QuadExprNode quadA)
        {
            return new Expr(new QuadExprNode()
            {
                LinearTerms = [.. quadA.LinearTerms],
                LinearWeights = quadA.LinearWeights.Select(w => w * b).ToList(),
                QuadraticTerms1 = [.. quadA.QuadraticTerms1],
                QuadraticTerms2 = [.. quadA.QuadraticTerms2],
                QuadraticWeights = quadA.QuadraticWeights.Select(w => w * b).ToList(),
                ConstantTerm = quadA.ConstantTerm * b
            });
        }
        if (na is ProductNode prodA)
        {
            return new Expr(new ProductNode([.. prodA.Factors])
            {
                Factor = prodA.Factor * b
            });
        }
        return new Expr(new ProductNode([na, new ConstantNode(b)]));
    }

    public static Expr operator *(double a, Expr b)
    {
        var nb = b._node;

        if (nb is LinExprNode linB)
        {
            return new Expr(new LinExprNode()
            {
                Terms = [.. linB.Terms],
                Weights = linB.Weights.Select(w => w * a).ToList(),
                ConstantTerm = linB.ConstantTerm * a
            });
        }
        if (nb is QuadExprNode quadB)
        {
            return new Expr(new QuadExprNode()
            {
                LinearTerms = [.. quadB.LinearTerms],
                LinearWeights = quadB.LinearWeights.Select(w => w * a).ToList(),
                QuadraticTerms1 = [.. quadB.QuadraticTerms1],
                QuadraticTerms2 = [.. quadB.QuadraticTerms2],
                QuadraticWeights = quadB.QuadraticWeights.Select(w => w * a).ToList(),
                ConstantTerm = quadB.ConstantTerm * a
            });
        }
        if (nb is ProductNode prodB)
        {
            return new Expr(new ProductNode([.. prodB.Factors])
            {
                Factor = prodB.Factor * a
            });
        }
        return new Expr(new ProductNode([new ConstantNode(a), nb]));
    }
    public static Expr operator /(Expr a, double b)
    {
        var na = a._node;

        if (na is LinExprNode linA)
        {
            return new Expr(new LinExprNode()
            {
                Terms = [.. linA.Terms],
                Weights = linA.Weights.Select(w => w / b).ToList(),
                ConstantTerm = linA.ConstantTerm / b
            });
        }
        if (na is QuadExprNode quadA)
        {
            return new Expr(new QuadExprNode()
            {
                LinearTerms = [.. quadA.LinearTerms],
                LinearWeights = quadA.LinearWeights.Select(w => w / b).ToList(),
                QuadraticTerms1 = [.. quadA.QuadraticTerms1],
                QuadraticTerms2 = [.. quadA.QuadraticTerms2],
                QuadraticWeights = quadA.QuadraticWeights.Select(w => w / b).ToList(),
                ConstantTerm = quadA.ConstantTerm / b
            });
        }
        return a * (1.0 / b);
    }
    public static Expr operator /(double a, Expr b) => new Expr(new DivisionNode(new ConstantNode(a), b._node));

    // C# 14 compound assignment operators - modify expression in-place for efficiency
    public void operator +=(Expr other)
    {
        var no = other._node;

        // Check if node is a QuadExpr - use efficient AddTerm
        if (_node is QuadExprNode quad)
        {
            quad.AddTerm(no, 1.0);
        }
        // Check if node is a LinExpr - use efficient AddTerm
        else if (_node is LinExprNode lin)
        {
            lin.AddTerm(no, 1.0);
        }
        // Check if this is a zero constant
        else if (_node is ConstantNode { Value: 0 })
        {
            _node = no;
        }
        // Otherwise create appropriate expression type
        else
        {
            var current = _node;
            // Use QuadExpr if either operand is quadratic
            if ((!current.IsLinear() && current.IsAtMostQuadratic()) ||
                (!no.IsLinear() && no.IsAtMostQuadratic()) ||
                current is QuadExprNode || no is QuadExprNode)
            {
                _node = new QuadExprNode([current, no]);
            }
            else
            {
                _node = new LinExprNode([current, no]);
            }
        }
    }

    public void operator -=(Expr other)
    {
        var no = other._node;

        // Check if node is a QuadExpr - use efficient AddTerm
        if (_node is QuadExprNode quad)
        {
            quad.AddTerm(no, -1.0);
        }
        // Check if node is a LinExpr - use efficient AddTerm
        else if (_node is LinExprNode lin)
        {
            lin.AddTerm(no, -1.0);
        }
        // Check if this is a zero constant
        else if (_node is ConstantNode { Value: 0 })
        {
            _node = new NegationNode(no);
        }
        // Otherwise create appropriate expression type
        else
        {
            var current = _node;
            // Use QuadExpr if either operand is quadratic
            if ((!current.IsLinear() && current.IsAtMostQuadratic()) ||
                (!no.IsLinear() && no.IsAtMostQuadratic()) ||
                current is QuadExprNode || no is QuadExprNode)
            {
                _node = new QuadExprNode([current, new NegationNode(no)]);
            }
            else
            {
                _node = new LinExprNode([current, new NegationNode(no)]);
            }
        }
    }

    public void operator *=(Expr other)
    {
        var no = other._node;

        // Special handling for LinExpr and QuadExpr multiplying by constant
        if (no is ConstantNode c)
        {
            if (_node is LinExprNode lin)
            {
                for (int i = 0; i < lin.Weights.Count; i++)
                    lin.Weights[i] *= c.Value;
                lin.ConstantTerm *= c.Value;
                return;
            }
            if (_node is QuadExprNode quad)
            {
                for (int i = 0; i < quad.LinearWeights.Count; i++)
                    quad.LinearWeights[i] *= c.Value;
                for (int i = 0; i < quad.QuadraticWeights.Count; i++)
                    quad.QuadraticWeights[i] *= c.Value;
                quad.ConstantTerm *= c.Value;
                return;
            }
        }

        // Check if node is a Product
        if (_node is ProductNode prod)
        {
            prod.Factors.Add(no);
        }
        // Check if this is a one constant
        else if (_node is ConstantNode { Value: 1 })
        {
            _node = no;
        }
        // Otherwise create a new Product with current value and new factor
        else
        {
            _node = new ProductNode([_node, no]);
        }
    }

    public void operator /=(Expr other)
    {
        var no = other._node;

        // Special handling for LinExpr and QuadExpr dividing by constant
        if (no is ConstantNode c)
        {
            if (_node is LinExprNode lin)
            {
                for (int i = 0; i < lin.Weights.Count; i++)
                    lin.Weights[i] /= c.Value;
                lin.ConstantTerm /= c.Value;
                return;
            }
            if (_node is QuadExprNode quad)
            {
                for (int i = 0; i < quad.LinearWeights.Count; i++)
                    quad.LinearWeights[i] /= c.Value;
                for (int i = 0; i < quad.QuadraticWeights.Count; i++)
                    quad.QuadraticWeights[i] /= c.Value;
                quad.ConstantTerm /= c.Value;
                return;
            }
        }

        // Check if node is a Product (add reciprocal)
        if (_node is ProductNode prod)
        {
            prod.Factors.Add(new DivisionNode(new ConstantNode(1), no));
        }
        // Otherwise create a division
        else
        {
            _node = new DivisionNode(_node, no);
        }
    }

    public Constraint Between(double lower, double upper) => new Constraint(this, lower, upper);

    /// <summary>
    /// Caches variables for this expression and all children to optimize repeated Hessian evaluations.
    /// Called once during model finalization before optimization.
    /// </summary>
    internal void Prepare()
    {
        _node.Prepare();
    }

    /// <summary>
    /// Clears cached variables to free memory after optimization completes.
    /// </summary>
    internal void Clear()
    {
        _node.Clear();
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

    public static Expr Pow(Expr @base, double exponent) => new Expr(new PowerOpNode(@base._node, exponent));
    public static Expr Pow(Expr @base, Expr exponent) => Exp(exponent * Log(@base));
    public static Expr Sqrt(Expr a) => new Expr(new PowerOpNode(a._node, 0.5));
    public static Expr Sin(Expr a) => new Expr(new SinNode(a._node));
    public static Expr Cos(Expr a) => new Expr(new CosNode(a._node));
    public static Expr Tan(Expr a) => new Expr(new TanNode(a._node));
    public static Expr Exp(Expr a) => new Expr(new ExpNode(a._node));
    public static Expr Log(Expr a) => new Expr(new LogNode(a._node));

    public override string ToString() => _node.ToString() ?? string.Empty;
}
