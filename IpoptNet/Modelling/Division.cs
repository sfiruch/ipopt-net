using System.Text;

namespace IpoptNet.Modelling;

public sealed class Division : Expr
{
    public Expr Left { get; set; }
    public Expr Right { get; set; }
    private double[]? _gradLBuffer;
    private double[]? _gradRBuffer;
    private int[]? _allVarsSorted;
    private int[]? _lIndices; // Index in _gradLBuffer for each var in _allVarsSorted, or -1 if not in L
    private int[]? _rIndices; // Index in _gradRBuffer for each var in _allVarsSorted, or -1 if not in R

    public Division(Expr left, Expr right)
    {
        Left = left;
        Right = right;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        return Left.Evaluate(x) / Right.Evaluate(x);
    }

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        var rVal = Right.Evaluate(x);
        var lVal = Left.Evaluate(x);
        Left.AccumulateGradientCompact(x, compactGrad, multiplier / rVal, sortedVarIndices);
        Right.AccumulateGradientCompact(x, compactGrad, -multiplier * lVal / (rVal * rVal), sortedVarIndices);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // f = l / r
        // df/dx = (l'r - lr') / r²
        // d²f/dx² = l''/r - lr''/r² - 2l'r'/r² + 2lr'²/r³
        var lVal = Left.Evaluate(x);
        var rVal = Right.Evaluate(x);
        var r2 = rVal * rVal;
        var r3 = r2 * rVal;

        Left.AccumulateHessian(x, hess, multiplier / rVal);
        Right.AccumulateHessian(x, hess, -multiplier * lVal / r2);

        // Compute compact gradients
        Array.Clear(_gradLBuffer!);
        Array.Clear(_gradRBuffer!);
        Left.AccumulateGradientCompact(x, _gradLBuffer!, 1.0, Left._sortedVarIndices!);
        Right.AccumulateGradientCompact(x, _gradRBuffer!, 1.0, Right._sortedVarIndices!);

        // Add 2lr'²/r³ (outer product of r' with itself)
        var coeffR = multiplier * 2 * lVal / r3;
        var sortedR = Right._sortedVarIndices!;
        for (int i = 0; i < sortedR.Length; i++)
        {
            var gI = _gradRBuffer![i];
            for (int j = 0; j <= i; j++)
                hess.Add(sortedR[i], sortedR[j], coeffR * gI * _gradRBuffer[j]);
        }

        // Add -(l'r' + r'l')/r² (cross terms between l' and r')
        var coeffCross = -multiplier / r2;
        // Iterate over pre-computed merged variable list (lower triangle only)
        for (int i = 0; i < _allVarsSorted!.Length; i++)
        {
            var varI = _allVarsSorted[i];
            var lIdxI = _lIndices![i];
            var rIdxI = _rIndices![i];

            // Get gradients at varI
            var gLi = lIdxI >= 0 ? _gradLBuffer![lIdxI] : 0.0;
            var gRi = rIdxI >= 0 ? _gradRBuffer![rIdxI] : 0.0;

            for (int j = 0; j <= i; j++)
            {
                var varJ = _allVarsSorted[j];
                var lIdxJ = _lIndices[j];
                var rIdxJ = _rIndices[j];

                // Get gradients at varJ
                var gLj = lIdxJ >= 0 ? _gradLBuffer![lIdxJ] : 0.0;
                var gRj = rIdxJ >= 0 ? _gradRBuffer![rIdxJ] : 0.0;

                // Add both terms of -(∂L/∂i * ∂R/∂j + ∂R/∂i * ∂L/∂j) / R²
                hess.Add(varI, varJ, coeffCross * (gLi * gRj + gRi * gLj));
            }
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
            foreach (var v in Left._cachedVariables!)
                vars.Add(v);
            foreach (var v in Right._cachedVariables!)
                vars.Add(v);
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

    protected override void PrepareChildren()
    {
        Left.Prepare();
        Right.Prepare();
        _gradLBuffer = new double[Left._cachedVariables!.Count];
        _gradRBuffer = new double[Right._cachedVariables!.Count];

        // Pre-compute merged variable list for cross-term computation
        var sortedL = Left._sortedVarIndices!;
        var sortedR = Right._sortedVarIndices!;

        // Merge two sorted arrays into one, keeping unique values
        var merged = new List<int>(sortedL.Length + sortedR.Length);
        var lIndicesList = new List<int>(sortedL.Length + sortedR.Length);
        var rIndicesList = new List<int>(sortedL.Length + sortedR.Length);

        int li = 0, ri = 0;
        while (li < sortedL.Length && ri < sortedR.Length)
        {
            if (sortedL[li] < sortedR[ri])
            {
                merged.Add(sortedL[li]);
                lIndicesList.Add(li);
                rIndicesList.Add(-1);
                li++;
            }
            else if (sortedL[li] > sortedR[ri])
            {
                merged.Add(sortedR[ri]);
                lIndicesList.Add(-1);
                rIndicesList.Add(ri);
                ri++;
            }
            else // equal
            {
                merged.Add(sortedL[li]);
                lIndicesList.Add(li);
                rIndicesList.Add(ri);
                li++;
                ri++;
            }
        }
        while (li < sortedL.Length)
        {
            merged.Add(sortedL[li]);
            lIndicesList.Add(li);
            rIndicesList.Add(-1);
            li++;
        }
        while (ri < sortedR.Length)
        {
            merged.Add(sortedR[ri]);
            lIndicesList.Add(-1);
            rIndicesList.Add(ri);
            ri++;
        }

        _allVarsSorted = merged.ToArray();
        _lIndices = lIndicesList.ToArray();
        _rIndices = rIndicesList.ToArray();
    }

    protected override void ClearChildren()
    {
        Left.Clear();
        Right.Clear();
        _gradLBuffer = null;
        _gradRBuffer = null;
        _allVarsSorted = null;
        _lIndices = null;
        _rIndices = null;
    }

    protected override string ToStringCore()
    {
        // If both operands are simple, format inline
        if (Left.IsSimpleForPrinting() && Right.IsSimpleForPrinting())
            return $"Division: {Left}/{Right}";

        // Otherwise, use multi-line tree format
        var sb = new StringBuilder();
        sb.AppendLine("Division:");
        sb.AppendLine("  Left:");
        foreach (var line in Left.ToString().Split(Environment.NewLine))
            sb.AppendLine($"    {line}");
        sb.AppendLine("  Right:");
        foreach (var line in Right.ToString().Split(Environment.NewLine))
            sb.AppendLine($"    {line}");
        return sb.ToString().TrimEnd();
    }
}
