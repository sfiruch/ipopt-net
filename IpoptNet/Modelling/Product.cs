using System.Text;

namespace IpoptNet.Modelling;

public sealed class Product : Expr
{
    public List<Expr> Factors { get; set; }
    public double Factor = 1.0;
    private double[][]? _factorGradBuffers;
    private double[]? _factorValues;
    private double[]? _excludingFactor;

    public Product() => Factors = [];
    public Product(List<Expr> factors)
    {
        // Extract all Constants and multiply them into Factor
        Factor = 1.0;
        Factors = [];
        foreach (var f in factors)
        {
            if (f is Constant c)
                Factor *= c.Value;
            else
                Factors.Add(f);
        }
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        if (Factor == 0.0)
            return 0.0;

        var result = Factor;
        foreach (var factor in Factors)
            result *= factor.Evaluate(x);
        return result;
    }

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        if (Factor == 0.0)
            return;

        var scaledMultiplier = multiplier * Factor;
        for (int i = 0; i < Factors.Count; i++)
        {
            var otherProduct = 1.0;
            for (int j = 0; j < Factors.Count; j++)
            {
                if (i != j)
                    otherProduct *= Factors[j].Evaluate(x);
            }
            Factors[i].AccumulateGradientCompact(x, compactGrad, scaledMultiplier * otherProduct, sortedVarIndices);
        }
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        if (Factor == 0.0)
            return;

        var scaledMultiplier = multiplier * Factor;

        if (Factors.Count == 0)
            return;
        if (Factors.Count == 1)
        {
            Factors[0].AccumulateHessian(x, hess, scaledMultiplier);
            return;
        }

        // Evaluate all factors once
        for (int i = 0; i < Factors.Count; i++)
            _factorValues![i] = Factors[i].Evaluate(x);

        // Pre-compute product excluding each factor
        var totalProduct = 1.0;
        for (int i = 0; i < Factors.Count; i++)
            totalProduct *= _factorValues![i];
        for (int i = 0; i < Factors.Count; i++)
            _excludingFactor![i] = totalProduct / _factorValues![i];

        // Compute compact gradients of all factors
        for (int i = 0; i < Factors.Count; i++)
            if (!Factors[i].IsConstantWrtX())
            {
                Array.Clear(_factorGradBuffers![i]);
                Factors[i].AccumulateGradientCompact(x, _factorGradBuffers[i], 1.0, Factors[i]._sortedVarIndices!);
            }

        // Accumulate Hessian from each factor's second derivative
        for (int k = 0; k < Factors.Count; k++)
            Factors[k].AccumulateHessian(x, hess, scaledMultiplier * _excludingFactor![k]);

        // Add cross terms between pairs of factors
        for (int k = 0; k + 1 < Factors.Count; k++)
        {
            if (Factors[k].IsConstantWrtX()) 
                continue;

            for (int m = k + 1; m < Factors.Count; m++)
            {
                if (Factors[m].IsConstantWrtX())
                    continue;

                AddCrossTermCompact(hess, _factorGradBuffers![k], _factorGradBuffers[m],
                    Factors[k]._sortedVarIndices!, Factors[m]._sortedVarIndices!, (double)(scaledMultiplier * (double)(_excludingFactor![k] / _factorValues![m])));
            }
        }
    }

    private static void AddCrossTermCompact(HessianAccumulator hess, double[] gradA, double[] gradB,
        int[] sortedA, int[] sortedB, double coeff)
    {
        for (int a = 0; a < sortedA.Length; a++)
        {
            var gAi = gradA[a];
            var gBi = 0.0;

            // Find if sortedA[a] exists in sortedB
            int bIndex = Array.BinarySearch(sortedB, sortedA[a]);
            if (bIndex >= 0)
                gBi = gradB[bIndex];

            for (int b = 0; b < sortedB.Length; b++)
            {
                var gBj = gradB[b];
                var gAj = 0.0;

                // Find if sortedB[b] exists in sortedA
                int aIndex = Array.BinarySearch(sortedA, sortedB[b]);
                if (aIndex >= 0)
                    gAj = gradA[aIndex];

                hess.Add(sortedA[a], sortedB[b], coeff * gAi * gBj);
                hess.Add(sortedB[b], sortedA[a], coeff * gBi * gAj);
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

        for (int i = 0; i < Factors.Count; i++)
        {
            for (int j = i + 1; j < Factors.Count; j++)
            {
                foreach (var v1 in Factors[i]._cachedVariables!)
                    foreach (var v2 in Factors[j]._cachedVariables!)
                        AddSparsityEntry(entries, v1.Index, v2.Index);
            }
        }
    }

    protected override bool IsConstantWrtXCore() => Factors.Count == 0 || Factors.All(f => f.IsConstantWrtX());

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

    protected override Expr CloneCore()
    {
        var clone = new Product([.. Factors]);
        clone.Factor = Factor;
        return clone;
    }

    protected override void PrepareChildren()
    {
        foreach (var factor in Factors)
            factor.Prepare();

        // Preallocate gradient buffers for each factor
        _factorGradBuffers = new double[Factors.Count][];
        for (int i = 0; i < Factors.Count; i++)
            _factorGradBuffers[i] = new double[Factors[i]._cachedVariables!.Count];

        // Preallocate factor evaluation and product arrays
        _factorValues = new double[Factors.Count];
        _excludingFactor = new double[Factors.Count];
    }

    protected override void ClearChildren()
    {
        foreach (var factor in Factors)
            factor.Clear();

        _factorGradBuffers = null;
        _factorValues = null;
        _excludingFactor = null;
    }

    protected override string ToStringCore()
    {
        // If all factors are simple, format inline
        if (Factors.All(f => f.IsSimpleForPrinting()))
        {
            var result = new StringBuilder();
            result.Append("Product: ");
            result.Append(Factor == 1.0 ? "" : Factor.ToString());
            foreach (var factor in Factors)
            {
                if (result.Length > 9) // Length of "Product: "
                    result.Append(" * ");
                result.Append($"({factor})");
            }
            return result.Length > 9 ? result.ToString() : "Product: 1";
        }

        // Otherwise, use multi-line tree format
        var sb = new StringBuilder();
        sb.AppendLine($"Product: {Factors.Count} factors, Factor={Factor}");
        for (int i = 0; i < Factors.Count; i++)
        {
            var factorLines = Factors[i].ToString().Split(Environment.NewLine);
            if (factorLines.Length == 1)
                sb.AppendLine($"  [{i}]: {factorLines[0]}");
            else
            {
                sb.AppendLine($"  [{i}]:");
                foreach (var line in factorLines)
                    sb.AppendLine($"    {line}");
            }
        }
        return sb.ToString().TrimEnd();
    }
}
