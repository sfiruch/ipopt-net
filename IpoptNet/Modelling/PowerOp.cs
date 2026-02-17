using System.Text;

namespace IpoptNet.Modelling;

internal sealed class PowerOpNode : ExprNode
{
    public ExprNode Base { get; set; }
    public double Exponent { get; set; }
    private double[]? _gradBuffer;

    public PowerOpNode(ExprNode @base, double exponent)
    {
        Base = @base;
        Exponent = exponent;
    }

    internal override double Evaluate(ReadOnlySpan<double> x) => Math.Pow(Base.Evaluate(x), Exponent);

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        var bVal = Base.Evaluate(x);
        var deriv = Exponent * Math.Pow(bVal, Exponent - 1);
        Base.AccumulateGradientCompact(x, compactGrad, multiplier * deriv, sortedVarIndices);
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // d²(b^n)/dx² = n*(n-1)*b^(n-2)*(db/dx)² + n*b^(n-1)*d²b/dx²
        var bVal = Base.Evaluate(x);
        var firstDerivCoeff = Exponent * Math.Pow(bVal, Exponent - 1);
        var secondDerivCoeff = Exponent * (Exponent - 1) * Math.Pow(bVal, Exponent - 2);
        Base.AccumulateHessian(x, hess, multiplier * firstDerivCoeff);

        // Outer product of gradient with itself using compact gradient
        var coeff = multiplier * secondDerivCoeff;

        Array.Clear(_gradBuffer!);
        Base.AccumulateGradientCompact(x, _gradBuffer!, 1.0, Base._sortedVarIndices!);

        var sorted = Base._sortedVarIndices!;
        for (int i = 0; i < sorted.Length; i++)
        {
            var gI = _gradBuffer![i];
            for (int j = 0; j <= i; j++)
                hess.Add(sorted[i], sorted[j], coeff * gI * _gradBuffer[j]);
        }
    }

    internal override void CollectVariables(HashSet<Variable> variables) => Base.CollectVariables(variables);

    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries)
    {
        if (!Base.IsConstantWrtX())
        {
            AddClique(entries, Base._cachedVariables!);
        }
    }

    internal override bool IsConstantWrtX() => Base.IsConstantWrtX();

    internal override bool IsLinear() => false;

    internal override bool IsAtMostQuadratic()
    {
        // At most quadratic if: (exponent is at most 2 and base is linear) or (exponent is 1 and base is quadratic) or (base is constant)
        if (Base.IsConstantWrtX())
            return true;
        if (Exponent == 2)
            return Base.IsLinear();
        return false;
    }

    internal override void PrepareChildren()
    {
        Base.Prepare();
        _gradBuffer = new double[Base._cachedVariables!.Count];
    }

    internal override void ClearChildren()
    {
        Base.Clear();
        _gradBuffer = null;
    }

    public override string ToString()
    {
        // If base is simple, format inline
        if (Base.IsSimpleForPrinting())
            return $"PowerOp: ({Base})^{Exponent}";

        // Otherwise, use multi-line tree format
        var sb = new StringBuilder();
        sb.AppendLine($"PowerOp: ^{Exponent}");
        foreach (var line in Base.ToString()!.Split(Environment.NewLine))
            sb.AppendLine($"  {line}");
        return sb.ToString().TrimEnd();
    }
}
