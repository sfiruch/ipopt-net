using System.Buffers;

namespace IpoptNet.Modelling;

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

        var varsL = Left._cachedVariables!;
        var varsR = Right._cachedVariables!;

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

    protected override void CacheVariablesForChildren()
    {
        Left.CacheVariables();
        Right.CacheVariables();
    }

    protected override void ClearCachedVariablesForChildren()
    {
        Left.ClearCachedVariables();
        Right.ClearCachedVariables();
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
