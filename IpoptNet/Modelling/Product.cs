using System.Buffers;

namespace IpoptNet.Modelling;

public sealed class Product : Expr
{
    public List<Expr> Factors { get; set; }
    public double Factor = 1.0;

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
        var result = Factor;
        foreach (var factor in Factors)
            result *= factor.Evaluate(x);
        return result;
    }

    protected override void AccumulateGradientCore(ReadOnlySpan<double> x, Span<double> grad, double multiplier)
    {
        // Product rule: d(f*g*h)/dx = df/dx*g*h + f*dg/dx*h + f*g*dh/dx
        var scaledMultiplier = multiplier * Factor;
        for (int i = 0; i < Factors.Count; i++)
        {
            var otherProduct = 1.0;
            for (int j = 0; j < Factors.Count; j++)
            {
                if (i != j)
                    otherProduct *= Factors[j].Evaluate(x);
            }
            Factors[i].AccumulateGradient(x, grad, scaledMultiplier * otherProduct);
        }
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        var scaledMultiplier = multiplier * Factor;

        if (Factors.Count == 0)
            return;
        if (Factors.Count == 1)
        {
            Factors[0].AccumulateHessian(x, hess, scaledMultiplier);
            return;
        }

        // Fast path for 2-factor products (most common case)
        if (Factors.Count == 2)
        {
            AccumulateHessian2Factors(x, hess, scaledMultiplier);
            return;
        }

        // Fast path for 3-factor products
        if (Factors.Count == 3)
        {
            AccumulateHessian3Factors(x, hess, scaledMultiplier);
            return;
        }

        var n = x.Length;

        // Evaluate all factors once
        var factorValues = ArrayPool<double>.Shared.Rent(Factors.Count);
        for (int i = 0; i < Factors.Count; i++)
            factorValues[i] = Factors[i].Evaluate(x);

        // Pre-compute product excluding each factor
        var excludingFactor = ArrayPool<double>.Shared.Rent(Factors.Count);
        var totalProduct = 1.0;
        for (int i = 0; i < Factors.Count; i++)
            totalProduct *= factorValues[i];
        for (int i = 0; i < Factors.Count; i++)
            excludingFactor[i] = totalProduct / factorValues[i];

        // Compute gradients of all factors once and keep track of non-zeros
        var factorGradients = new double[Factors.Count][];
        var nonZeroIndices = new int[Factors.Count][];
        var nonZeroCounts = new int[Factors.Count];

        for (int i = 0; i < Factors.Count; i++)
        {
            if (Factors[i].IsConstantWrtX())
            {
                nonZeroIndices[i] = Array.Empty<int>();
                nonZeroCounts[i] = 0;
                continue;
            }

            // 1. Use cached variables
            var vars = Factors[i]._cachedVariables!;

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
            var nonZerosList = new List<int>(vars.Count);
            foreach (var v in vars)
            {
                if (Math.Abs(rented[v.Index]) > 1e-18)
                    nonZerosList.Add(v.Index);
            }
            nonZeroIndices[i] = nonZerosList.ToArray();
            nonZeroCounts[i] = nonZerosList.Count;
        }

        // 1. Accumulate Hessian from each factor's second derivative
        // Use pre-computed excludingFactor
        for (int k = 0; k < Factors.Count; k++)
        {
            var otherProduct = excludingFactor[k];
            if (Math.Abs(otherProduct) > 1e-18)
                Factors[k].AccumulateHessian(x, hess, scaledMultiplier * otherProduct);
        }

        // 2. Add cross terms between pairs of factors
        for (int k = 0; k < Factors.Count; k++)
        {
            var idxK = nonZeroIndices[k];
            if (idxK == null || nonZeroCounts[k] == 0) continue;
            var gradK = factorGradients[k];
            var spanK = idxK.AsSpan(0, nonZeroCounts[k]);

            for (int m = k + 1; m < Factors.Count; m++)
            {
                var idxM = nonZeroIndices[m];
                if (idxM == null || nonZeroCounts[m] == 0) continue;

                // Use pre-computed product
                var otherProduct = excludingFactor[k] / factorValues[m];

                if (Math.Abs(otherProduct) > 1e-18)
                {
                    var coeff = scaledMultiplier * otherProduct;
                    var gradM = factorGradients[m];
                    var spanM = idxM.AsSpan(0, nonZeroCounts[m]);

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

        ArrayPool<double>.Shared.Return(factorValues);
        ArrayPool<double>.Shared.Return(excludingFactor);
    }

    private void AccumulateHessian2Factors(ReadOnlySpan<double> x, HessianAccumulator hess, double scaledMultiplier)
    {
        // Optimized fast path for 2-factor products: f = F0 * F1

        var val0 = Factors[0].Evaluate(x);
        var val1 = Factors[1].Evaluate(x);

        // Hessian contributions from each factor's second derivative
        if (!Factors[0].IsConstantWrtX() && Math.Abs(val1) > 1e-18)
            Factors[0].AccumulateHessian(x, hess, scaledMultiplier * val1);
        if (!Factors[1].IsConstantWrtX() && Math.Abs(val0) > 1e-18)
            Factors[1].AccumulateHessian(x, hess, scaledMultiplier * val0);

        // Cross term: outer product of gradients
        if (!Factors[0].IsConstantWrtX() && !Factors[1].IsConstantWrtX())
        {
            var vars0 = Factors[0]._cachedVariables!;
            var vars1 = Factors[1]._cachedVariables!;
            var n = x.Length;

            var grad0 = ArrayPool<double>.Shared.Rent(n);
            var grad1 = ArrayPool<double>.Shared.Rent(n);

            if (vars0.Count < n / 32)
                foreach (var v in vars0)
                    grad0[v.Index] = 0.0;
            else
                Array.Clear(grad0, 0, n);

            if (vars1.Count < n / 32)
                foreach (var v in vars1)
                    grad1[v.Index] = 0.0;
            else
                Array.Clear(grad1, 0, n);

            Factors[0].AccumulateGradient(x, grad0, 1.0);
            Factors[1].AccumulateGradient(x, grad1, 1.0);

            AddCrossTerm(hess, grad0, grad1, vars0, vars1, scaledMultiplier);

            ArrayPool<double>.Shared.Return(grad0);
            ArrayPool<double>.Shared.Return(grad1);
        }
    }

    private void AccumulateHessian3Factors(ReadOnlySpan<double> x, HessianAccumulator hess, double scaledMultiplier)
    {
        // Optimized fast path for 3-factor products: f = F0 * F1 * F2

        var val0 = Factors[0].Evaluate(x);
        var val1 = Factors[1].Evaluate(x);
        var val2 = Factors[2].Evaluate(x);

        // Hessian contributions from each factor's second derivative
        var product012 = val0 * val1 * val2;
        if (!Factors[0].IsConstantWrtX() && Math.Abs(product012 / val0) > 1e-18)
            Factors[0].AccumulateHessian(x, hess, scaledMultiplier * val1 * val2);
        if (!Factors[1].IsConstantWrtX() && Math.Abs(product012 / val1) > 1e-18)
            Factors[1].AccumulateHessian(x, hess, scaledMultiplier * val0 * val2);
        if (!Factors[2].IsConstantWrtX() && Math.Abs(product012 / val2) > 1e-18)
            Factors[2].AccumulateHessian(x, hess, scaledMultiplier * val0 * val1);

        // Compute gradients for non-constant factors
        var n = x.Length;
        var grad0 = !Factors[0].IsConstantWrtX() ? ArrayPool<double>.Shared.Rent(n) : null;
        var grad1 = !Factors[1].IsConstantWrtX() ? ArrayPool<double>.Shared.Rent(n) : null;
        var grad2 = !Factors[2].IsConstantWrtX() ? ArrayPool<double>.Shared.Rent(n) : null;

        if (grad0 != null)
        {
            var vars0 = Factors[0]._cachedVariables!;
            if (vars0.Count < n / 32)
                foreach (var v in vars0)
                    grad0[v.Index] = 0.0;
            else
                Array.Clear(grad0, 0, n);
            Factors[0].AccumulateGradient(x, grad0, 1.0);
        }

        if (grad1 != null)
        {
            var vars1 = Factors[1]._cachedVariables!;
            if (vars1.Count < n / 32)
                foreach (var v in vars1)
                    grad1[v.Index] = 0.0;
            else
                Array.Clear(grad1, 0, n);
            Factors[1].AccumulateGradient(x, grad1, 1.0);
        }

        if (grad2 != null)
        {
            var vars2 = Factors[2]._cachedVariables!;
            if (vars2.Count < n / 32)
                foreach (var v in vars2)
                    grad2[v.Index] = 0.0;
            else
                Array.Clear(grad2, 0, n);
            Factors[2].AccumulateGradient(x, grad2, 1.0);
        }

        // Cross terms between pairs of factors
        // Cross term 0-1
        if (grad0 != null && grad1 != null)
        {
            var coeff = scaledMultiplier * val2;
            if (Math.Abs(coeff) > 1e-18)
                AddCrossTerm(hess, grad0, grad1, Factors[0]._cachedVariables!, Factors[1]._cachedVariables!, coeff);
        }

        // Cross term 0-2
        if (grad0 != null && grad2 != null)
        {
            var coeff = scaledMultiplier * val1;
            if (Math.Abs(coeff) > 1e-18)
                AddCrossTerm(hess, grad0, grad2, Factors[0]._cachedVariables!, Factors[2]._cachedVariables!, coeff);
        }

        // Cross term 1-2
        if (grad1 != null && grad2 != null)
        {
            var coeff = scaledMultiplier * val0;
            if (Math.Abs(coeff) > 1e-18)
                AddCrossTerm(hess, grad1, grad2, Factors[1]._cachedVariables!, Factors[2]._cachedVariables!, coeff);
        }

        // Return rented arrays
        if (grad0 != null) ArrayPool<double>.Shared.Return(grad0);
        if (grad1 != null) ArrayPool<double>.Shared.Return(grad1);
        if (grad2 != null) ArrayPool<double>.Shared.Return(grad2);
    }

    private static void AddCrossTerm(HessianAccumulator hess, double[] gradA, double[] gradB,
        HashSet<Variable> varsA, HashSet<Variable> varsB, double coeff)
    {
        foreach (var vA in varsA)
        {
            var gAi = gradA[vA.Index];
            var gBi = gradB[vA.Index];
            var c_gAi = coeff * gAi;
            var c_gBi = coeff * gBi;

            foreach (var vB in varsB)
            {
                var gBj = gradB[vB.Index];
                var gAj = gradA[vB.Index];

                // Add both parts of symmetric outer product
                hess.Add(vA.Index, vB.Index, c_gAi * gBj);
                hess.Add(vB.Index, vA.Index, c_gBi * gAj);
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

    protected override void CacheVariablesForChildren()
    {
        foreach (var factor in Factors)
            factor.CacheVariables();
    }

    protected override void ClearCachedVariablesForChildren()
    {
        foreach (var factor in Factors)
            factor.ClearCachedVariables();
    }

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}Product: {Factors.Count} factors, Factor={Factor}");
        for (int i = 0; i < Factors.Count; i++)
        {
            writer.WriteLine($"{indent}  [{i}]:");
            Factors[i].Print(writer, indent + "    ");
        }
    }
}
