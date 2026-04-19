using System.Text;

namespace IpoptNet.Modelling;

internal sealed class ProductNode : ExprNode
{
    public List<ExprNode> Factors { get; set; }
    public double Factor = 1.0;
    private double[][]? _factorGradBuffers;
    private double[]? _factorValues;
    private double[]? _excludingFactor;

    public ProductNode() => Factors = [];
    public ProductNode(List<ExprNode> factors)
    {
        // Extract all Constants and multiply them into Factor
        Factor = 1.0;
        Factors = [];
        foreach (var f in factors)
        {
            if (f is ConstantNode c)
                Factor *= c.Value;
            else
                Factors.Add(f);
        }
    }

    internal override double Evaluate(ReadOnlySpan<double> x)
    {
        if (Factor == 0.0)
            return 0.0;

        var result = Factor;
        foreach (var factor in Factors)
            result *= factor.Evaluate(x);
        return result;
    }

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        if (Factor == 0.0)
            return;

        var scaledMultiplier = multiplier * Factor;

        // Evaluate all factors once
        for (int i = 0; i < Factors.Count; i++)
            _factorValues![i] = Factors[i].Evaluate(x);

        for (int i = 0; i < Factors.Count; i++)
        {
            var otherProduct = 1.0;
            for (int j = 0; j < Factors.Count; j++)
                if (i != j)
                    otherProduct *= _factorValues![j];
            Factors[i].AccumulateGradientCompact(x, compactGrad, scaledMultiplier * otherProduct, sortedVarIndices);
        }
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
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

        // Evaluate all factors once, and summarize the zero structure so we can answer
        // "product of all factors except {i}" and "... except {k, m}" in O(1) without dividing
        // by an evaluated factor value (which yields NaN when that factor happens to be zero).
        var nonzeroProduct = 1.0;
        var zeroCount = 0;
        var zeroIdx1 = -1;
        var zeroIdx2 = -1;
        for (int i = 0; i < Factors.Count; i++)
        {
            var v = Factors[i].Evaluate(x);
            _factorValues![i] = v;
            if (v == 0.0)
            {
                if (zeroCount == 0) zeroIdx1 = i;
                else if (zeroCount == 1) zeroIdx2 = i;
                zeroCount++;
            }
            else
                nonzeroProduct *= v;
        }

        // _excludingFactor[i] = product of all factors except i.
        // - 0 zeros:   nonzeroProduct / _factorValues[i] (safe: denominator nonzero).
        // - 1 zero:    if i == that zero's index, result is nonzeroProduct; else the zero is still
        //              in the product → 0.
        // - ≥2 zeros:  at least one zero is always retained → 0.
        for (int i = 0; i < Factors.Count; i++)
        {
            _excludingFactor![i] = zeroCount switch
            {
                0 => nonzeroProduct / _factorValues![i],
                1 => i == zeroIdx1 ? nonzeroProduct : 0.0,
                _ => 0.0,
            };
        }

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

        // Add cross terms between pairs of factors. The coefficient is the product of all factors
        // except k and m, computed in O(1) from the nonzero summary (see _excludingFactor above).
        // - 0 zeros:   nonzeroProduct / (factor[k] * factor[m]).
        // - 1 zero:    nonzero only when that zero is k or m — then divide by the other factor.
        // - 2 zeros:   nonzero only when the pair is exactly those two zero indices.
        // - ≥3 zeros:  always zero.
        for (int k = 0; k + 1 < Factors.Count; k++)
        {
            if (Factors[k].IsConstantWrtX())
                continue;

            for (int m = k + 1; m < Factors.Count; m++)
            {
                if (Factors[m].IsConstantWrtX())
                    continue;

                var pairCoeff = zeroCount switch
                {
                    0 => nonzeroProduct / (_factorValues![k] * _factorValues![m]),
                    1 when k == zeroIdx1 => nonzeroProduct / _factorValues![m],
                    1 when m == zeroIdx1 => nonzeroProduct / _factorValues![k],
                    2 when (k == zeroIdx1 && m == zeroIdx2) || (k == zeroIdx2 && m == zeroIdx1) => nonzeroProduct,
                    _ => 0.0,
                };

                AddCrossTermCompact(hess, _factorGradBuffers![k], _factorGradBuffers[m],
                    Factors[k]._sortedVarIndices!, Factors[m]._sortedVarIndices!, scaledMultiplier * pairCoeff);
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

    internal override void CollectVariables(HashSet<Variable> variables)
    {
        foreach (var factor in Factors)
            factor.CollectVariables(variables);
    }

    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries)
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

    internal override bool IsConstantWrtX() => Factors.Count == 0 || Factors.All(f => f.IsConstantWrtX());

    internal override bool IsLinear()
    {
        // Linear if at most one factor is non-constant and that factor is linear
        var nonConstantFactors = Factors.Where(f => !f.IsConstantWrtX()).ToList();
        return nonConstantFactors.Count == 0 || (nonConstantFactors.Count == 1 && nonConstantFactors[0].IsLinear());
    }

    internal override bool IsAtMostQuadratic()
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

    internal override void PrepareChildren()
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

    internal override void ClearChildren()
    {
        foreach (var factor in Factors)
            factor.Clear();

        _factorGradBuffers = null;
        _factorValues = null;
        _excludingFactor = null;
    }

    public override string ToString()
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
            var factorLines = Factors[i].ToString()!.Split(Environment.NewLine);
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
