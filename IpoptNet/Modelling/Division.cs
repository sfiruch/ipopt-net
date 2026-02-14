namespace IpoptNet.Modelling;

public sealed class Division : Expr
{
    public Expr Left { get; set; }
    public Expr Right { get; set; }
    private double[]? _gradLBuffer;
    private double[]? _gradRBuffer;

    public Division(Expr left, Expr right)
    {
        Left = left;
        Right = right;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        return Left.Evaluate(x) / Right.Evaluate(x);
    }

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, Dictionary<int, int> varIndexToCompact)
    {
        var rVal = Right.Evaluate(x);
        var lVal = Left.Evaluate(x);
        Left.AccumulateGradientCompact(x, compactGrad, multiplier / rVal, varIndexToCompact);
        Right.AccumulateGradientCompact(x, compactGrad, -multiplier * lVal / (rVal * rVal), varIndexToCompact);
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

        if (Math.Abs(multiplier) < 1e-18) return;

        Left.AccumulateHessian(x, hess, multiplier / rVal);
        Right.AccumulateHessian(x, hess, -multiplier * lVal / r2);

        // Compute compact gradients
        Array.Clear(_gradLBuffer!);
        Array.Clear(_gradRBuffer!);
        Left.AccumulateGradientCompact(x, _gradLBuffer!, 1.0, Left._varIndexToCompact!);
        Right.AccumulateGradientCompact(x, _gradRBuffer!, 1.0, Right._varIndexToCompact!);

        // Add 2lr'²/r³ (outer product of r' with itself)
        var coeffR = multiplier * 2 * lVal / r3;
        if (Math.Abs(coeffR) > 1e-18)
        {
            var sortedR = Right._sortedVarIndices!;
            for (int i = 0; i < sortedR.Length; i++)
            {
                var gI = _gradRBuffer![i];
                for (int j = 0; j <= i; j++)
                    hess.Add(sortedR[i], sortedR[j], coeffR * gI * _gradRBuffer[j]);
            }
        }

        // Add -2l'r'/r² (cross terms between l' and r')
        var coeffCross = -2 * multiplier / r2;
        if (Math.Abs(coeffCross) > 1e-18)
        {
            var sortedL = Left._sortedVarIndices!;
            var sortedR = Right._sortedVarIndices!;

            for (int i = 0; i < sortedL.Length; i++)
            {
                var gLi = _gradLBuffer![i];
                var idxI = sortedL[i];

                for (int j = 0; j < sortedR.Length; j++)
                {
                    var idxJ = sortedR[j];
                    // Add symmetric contribution (divide by 2 since we add both directions)
                    hess.Add(idxI, idxJ, coeffCross / 2 * gLi * _gradRBuffer![j]);
                }
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
    }

    protected override void ClearChildren()
    {
        Left.Clear();
        Right.Clear();
        _gradLBuffer = null;
        _gradRBuffer = null;
    }

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}Division:");
        writer.WriteLine($"{indent}  Left:");
        Left.Print(writer, indent + "    ");
        writer.WriteLine($"{indent}  Right:");
        Right.Print(writer, indent + "    ");
    }
}
