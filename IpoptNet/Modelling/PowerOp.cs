namespace IpoptNet.Modelling;

public sealed class PowerOp : Expr
{
    public Expr Base { get; set; }
    public double Exponent { get; set; }
    private double[]? _gradBuffer;

    public PowerOp(Expr @base, double exponent)
    {
        Base = @base;
        Exponent = exponent;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Pow(Base.Evaluate(x), Exponent);

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, Dictionary<int, int> varIndexToCompact)
    {
        var bVal = Base.Evaluate(x);
        var deriv = Exponent * Math.Pow(bVal, Exponent - 1);
        Base.AccumulateGradientCompact(x, compactGrad, multiplier * deriv, varIndexToCompact);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // d²(b^n)/dx² = n*(n-1)*b^(n-2)*(db/dx)² + n*b^(n-1)*d²b/dx²
        var bVal = Base.Evaluate(x);
        var firstDerivCoeff = Exponent * Math.Pow(bVal, Exponent - 1);
        var secondDerivCoeff = Exponent * (Exponent - 1) * Math.Pow(bVal, Exponent - 2);
        Base.AccumulateHessian(x, hess, multiplier * firstDerivCoeff);

        // Outer product of gradient with itself using compact gradient
        var coeff = multiplier * secondDerivCoeff;
        if (Math.Abs(coeff) < 1e-18) return;

        Array.Clear(_gradBuffer!);
        Base.AccumulateGradientCompact(x, _gradBuffer!, 1.0, Base._varIndexToCompact!);

        var sorted = Base._sortedVarIndices!;
        for (int i = 0; i < sorted.Length; i++)
        {
            var gI = _gradBuffer![i];
            for (int j = 0; j <= i; j++)
                hess.Add(sorted[i], sorted[j], coeff * gI * _gradBuffer[j]);
        }
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
            AddClique(entries, Base._cachedVariables!);
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

    protected override void PrepareChildren()
    {
        Base.Prepare();
        _gradBuffer = new double[Base._cachedVariables!.Count];
    }

    protected override void ClearChildren()
    {
        Base.Clear();
        _gradBuffer = null;
    }

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}PowerOp: ^{Exponent}");
        Base.Print(writer, indent + "  ");
    }
}
