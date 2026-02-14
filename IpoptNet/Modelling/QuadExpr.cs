using System.ComponentModel.Design;
using System.Text;

namespace IpoptNet.Modelling;

public class QuadExpr : Expr
{
    public List<Expr> LinearTerms { get; set; }
    public List<double> LinearWeights { get; set; }
    public List<Expr> QuadraticTerms1 { get; set; }
    public List<Expr> QuadraticTerms2 { get; set; }
    public List<double> QuadraticWeights { get; set; }
    public double ConstantTerm { get; set; }

    public QuadExpr()
    {
        LinearTerms = [];
        LinearWeights = [];
        QuadraticTerms1 = [];
        QuadraticTerms2 = [];
        QuadraticWeights = [];
        ConstantTerm = 0.0;
    }

    /// <summary>
    /// Efficiently adds a term to this QuadExpr with proper Product expansion.
    /// Used by += operator for O(1) appending instead of O(n) copying.
    /// </summary>
    public void AddTerm(Expr term, double weight = 1.0)
    {
        var constantSum = ConstantTerm;
        ProcessTerm(term, weight, ref constantSum, LinearTerms, LinearWeights, QuadraticTerms1, QuadraticTerms2, QuadraticWeights);
        ConstantTerm = constantSum;
    }

    public QuadExpr(List<Expr> terms)
    {
        var constantSum = 0.0;
        var linearTerms = new List<Expr>();
        var linearWeights = new List<double>();
        var quadTerms1 = new List<Expr>();
        var quadTerms2 = new List<Expr>();
        var quadWeights = new List<double>();

        foreach (var term in terms)
        {
            if (term is Constant c)
            {
                constantSum += c.Value;
            }
            else if (term is Negation neg)
            {
                ProcessTerm(neg.Operand, -1.0, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
            }
            else if (term is LinExpr lin)
            {
                // Merge LinExpr - but check each term to see if it's actually quadratic
                constantSum += lin.ConstantTerm;
                for (int i = 0; i < lin.Terms.Count; i++)
                {
                    var linTerm = lin.Terms[i];
                    var linWeight = lin.Weights[i];

                    // Check if this term is actually quadratic (shouldn't be in LinExpr!)
                    if (linTerm is Product prod)
                    {
                        // Process the product with the weight from LinExpr
                        ProcessProduct(prod, linWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                    }
                    else if (linTerm is PowerOp { Exponent: 2 } pow)
                    {
                        // x^2 term with weight
                        quadTerms1.Add(pow.Base);
                        quadTerms2.Add(pow.Base);
                        quadWeights.Add(linWeight);
                    }
                    else
                    {
                        // Actually linear
                        linearTerms.Add(linTerm);
                        linearWeights.Add(linWeight);
                    }
                }
            }
            else if (term is QuadExpr quad)
            {
                // Merge QuadExpr - must check linear terms for Products!
                constantSum += quad.ConstantTerm;
                for (int i = 0; i < quad.LinearTerms.Count; i++)
                {
                    var quadLinTerm = quad.LinearTerms[i];
                    var quadLinWeight = quad.LinearWeights[i];

                    // Check if this linear term is actually quadratic (shouldn't be, but might be from old code)
                    if (quadLinTerm is Product prod)
                    {
                        ProcessProduct(prod, quadLinWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                    }
                    else if (quadLinTerm is PowerOp { Exponent: 2 } pow)
                    {
                        quadTerms1.Add(pow.Base);
                        quadTerms2.Add(pow.Base);
                        quadWeights.Add(quadLinWeight);
                    }
                    else
                    {
                        linearTerms.Add(quadLinTerm);
                        linearWeights.Add(quadLinWeight);
                    }
                }
                for (int i = 0; i < quad.QuadraticTerms1.Count; i++)
                {
                    var quadWeight = quad.QuadraticWeights[i];
                    quadTerms1.Add(quad.QuadraticTerms1[i]);
                    quadTerms2.Add(quad.QuadraticTerms2[i]);
                    quadWeights.Add(quadWeight);
                }
            }
            else if (term is Product prod)
            {
                ProcessProduct(prod, 1.0, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
            }
            else if (term is PowerOp { Exponent: 2 } pow)
            {
                // x^2 → quadratic term
                // But if base is a LinExpr, we need to expand it: (a*x + b*y + c)^2
                if (pow.Base is LinExpr linBase)
                {
                    // Expand (LinExpr)^2 into quadratic and linear terms
                    // (a1*t1 + a2*t2 + ... + c)^2 = sum(ai*aj*ti*tj) + 2*sum(ai*ti*c) + c^2

                    // Quadratic cross terms: ai * aj * ti * tj for all i,j pairs
                    for (int i = 0; i < linBase.Terms.Count; i++)
                    {
                        for (int j = i; j < linBase.Terms.Count; j++)
                        {
                            double weight = linBase.Weights[i] * linBase.Weights[j];
                            if (i != j) weight *= 2.0; // Cross terms appear twice

                            quadTerms1.Add(linBase.Terms[i]);
                            quadTerms2.Add(linBase.Terms[j]);
                            quadWeights.Add(weight);
                        }
                    }

                    // Linear terms from 2 * c * sum(ai*ti)
                    if (linBase.ConstantTerm != 0.0)
                    {
                        for (int i = 0; i < linBase.Terms.Count; i++)
                        {
                            var linearWeight = 2.0 * linBase.ConstantTerm * linBase.Weights[i];
                            linearTerms.Add(linBase.Terms[i]);
                            linearWeights.Add(linearWeight);
                        }
                    }

                    // Constant term: c^2
                    constantSum += linBase.ConstantTerm * linBase.ConstantTerm;
                }
                else
                {
                    // Simple case: just a variable or other expression squared
                    quadTerms1.Add(pow.Base);
                    quadTerms2.Add(pow.Base);
                    quadWeights.Add(1.0);
                }
            }
            else
            {
                // Fallback: add as linear term
                linearTerms.Add(term);
                linearWeights.Add(1.0);
            }
        }

        LinearTerms = linearTerms;
        LinearWeights = linearWeights;
        QuadraticTerms1 = quadTerms1;
        QuadraticTerms2 = quadTerms2;
        QuadraticWeights = quadWeights;
        ConstantTerm = constantSum;
    }

    private static void ProcessTerm(Expr term, double weight, ref double constantSum,
        List<Expr> linearTerms, List<double> linearWeights,
        List<Expr> quadTerms1, List<Expr> quadTerms2, List<double> quadWeights)
    {
        // Skip terms with zero weight
        if (weight == 0)
            return;

        if (term is Constant c)
            constantSum += weight * c.Value;
        else if (term is Negation neg)
            ProcessTerm(neg.Operand, -weight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
        else if (term is LinExpr lin)
        {
            constantSum += weight * lin.ConstantTerm;
            for (int i = 0; i < lin.Terms.Count; i++)
            {
                // Don't just copy - recursively process each term in case it's a Product
                var linTerm = lin.Terms[i];
                var scaledWeight = weight * lin.Weights[i];

                if (linTerm is Product prod)
                {
                    ProcessProduct(prod, scaledWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                }
                else if (linTerm is PowerOp { Exponent: 2 } pow)
                {
                    quadTerms1.Add(pow.Base);
                    quadTerms2.Add(pow.Base);
                    quadWeights.Add(scaledWeight);
                }
                else
                {
                    linearTerms.Add(linTerm);
                    linearWeights.Add(scaledWeight);
                }
            }
        }
        else if (term is QuadExpr quad)
        {
            constantSum += weight * quad.ConstantTerm;
            for (int i = 0; i < quad.LinearTerms.Count; i++)
            {
                // Don't just copy - recursively process each linear term in case it's a Product
                var quadLinTerm = quad.LinearTerms[i];
                var scaledWeight = weight * quad.LinearWeights[i];

                if (quadLinTerm is Product prod)
                    ProcessProduct(prod, scaledWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                else if (quadLinTerm is PowerOp { Exponent: 2 } pow)
                {
                    quadTerms1.Add(pow.Base);
                    quadTerms2.Add(pow.Base);
                    quadWeights.Add(scaledWeight);
                }
                else
                {
                    linearTerms.Add(quadLinTerm);
                    linearWeights.Add(scaledWeight);
                }
            }
            for (int i = 0; i < quad.QuadraticTerms1.Count; i++)
            {
                var scaledWeight = weight * quad.QuadraticWeights[i];
                quadTerms1.Add(quad.QuadraticTerms1[i]);
                quadTerms2.Add(quad.QuadraticTerms2[i]);
                quadWeights.Add(scaledWeight);
            }
        }
        else if (term is Product prod)
        {
            ProcessProduct(prod, weight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
        }
        else if (term is PowerOp { Exponent: 2 } pow)
        {
            // x^2 → quadratic term
            // But if base is a LinExpr, we need to expand it: (a*x + b*y + c)^2
            if (pow.Base is LinExpr linBase)
            {
                // Expand (LinExpr)^2 into quadratic and linear terms
                // (a1*t1 + a2*t2 + ... + c)^2 = sum(ai*aj*ti*tj) + 2*sum(ai*ti*c) + c^2

                // Quadratic cross terms: ai * aj * ti * tj for all i,j pairs
                for (int i = 0; i < linBase.Terms.Count; i++)
                {
                    for (int j = i; j < linBase.Terms.Count; j++)
                    {
                        double termWeight = weight * linBase.Weights[i] * linBase.Weights[j];
                        if (i != j) termWeight *= 2.0; // Cross terms appear twice

                        quadTerms1.Add(linBase.Terms[i]);
                        quadTerms2.Add(linBase.Terms[j]);
                        quadWeights.Add(termWeight);
                    }
                }

                // Linear terms from 2 * c * sum(ai*ti)
                if (linBase.ConstantTerm != 0.0)
                    for (int i = 0; i < linBase.Terms.Count; i++)
                    {
                        linearTerms.Add(linBase.Terms[i]);
                        linearWeights.Add((double)(weight * 2.0 * linBase.ConstantTerm * linBase.Weights[i]));
                    }

                // Constant term: c^2
                constantSum += weight * linBase.ConstantTerm * linBase.ConstantTerm;
            }
            else
            {
                // Simple case: just a variable or other expression squared
                quadTerms1.Add(pow.Base);
                quadTerms2.Add(pow.Base);
                quadWeights.Add(weight);
            }
        }
        else
        {
            linearTerms.Add(term);
            linearWeights.Add(weight);
        }
    }

    private static void ProcessProduct(Product prod, double weight, ref double constantSum,
        List<Expr> linearTerms, List<double> linearWeights,
        List<Expr> quadTerms1, List<Expr> quadTerms2, List<double> quadWeights)
    {
        // Product now stores constants in Factor field
        var productWeight = weight * prod.Factor;

        if (productWeight == 0)
            return;

        // Handle based on number of non-constant factors
        if (prod.Factors.Count == 0)
        {
            // All constants
            constantSum += productWeight;
        }
        else if (prod.Factors.Count == 1)
        {
            // Linear term: c * expr
            var expr = prod.Factors[0];
            if (expr is PowerOp { Exponent: 2 } pow)
            {
                // c * x^2
                quadTerms1.Add(pow.Base);
                quadTerms2.Add(pow.Base);
                quadWeights.Add(productWeight);
            }
            else
            {
                ProcessTerm(expr, productWeight, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
            }
        }
        else if (prod.Factors.Count == 2)
        {
            // Bilinear or quadratic term: c * expr1 * expr2
            var expr1 = prod.Factors[0];
            var expr2 = prod.Factors[1];

            // Check for x^2 represented as x*x
            if (expr1 is Variable v1 && expr2 is Variable v2 && v1.Index == v2.Index)
            {
                quadTerms1.Add(expr1);
                quadTerms2.Add(expr1);
                quadWeights.Add(productWeight);
            }
            // Expand LinExpr * LinExpr
            else if (expr1 is LinExpr lin1 && expr2 is LinExpr lin2)
            {
                // Expand: (Σ a_i * x_i + c1) * (Σ b_j * y_j + c2)
                // = Σ_i Σ_j (a_i * b_j * x_i * y_j) + Σ_i (a_i * c2 * x_i) + Σ_j (b_j * c1 * y_j) + c1 * c2

                // Constant * Constant
                constantSum += productWeight * lin1.ConstantTerm * lin2.ConstantTerm;

                // Linear terms from first LinExpr * constant from second
                for (int i = 0; i < lin1.Terms.Count; i++)
                {
                    var termWeight = productWeight * lin1.Weights[i] * lin2.ConstantTerm;
                    // Don't just add the term - process it in case it's a Product
                    ProcessTerm(lin1.Terms[i], termWeight,
                        ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                }

                // Linear terms from second LinExpr * constant from first
                for (int j = 0; j < lin2.Terms.Count; j++)
                {
                    var termWeight = productWeight * lin2.Weights[j] * lin1.ConstantTerm;
                    // Don't just add the term - process it in case it's a Product
                    ProcessTerm(lin2.Terms[j], termWeight,
                        ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);
                }

                // Quadratic cross terms
                for (int i = 0; i < lin1.Terms.Count; i++)
                    for (int j = 0; j < lin2.Terms.Count; j++)
                    {
                        quadTerms1.Add(lin1.Terms[i]);
                        quadTerms2.Add(lin2.Terms[j]);
                        quadWeights.Add((double)(productWeight * lin1.Weights[i] * lin2.Weights[j]));
                    }
            }
            // Expand LinExpr * other
            else if (expr1 is LinExpr lin)
            {
                // (Σ a_i * x_i + c) * y = Σ (a_i * x_i * y) + c * y
                // Add c * y as a linear or quadratic term depending on y
                if (lin.ConstantTerm != 0.0)
                    ProcessTerm(expr2, (double)(productWeight * lin.ConstantTerm), ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);

                for (int i = 0; i < lin.Terms.Count; i++)
                {
                    // Create bilinear term, but the term from LinExpr might itself be a Product
                    // so we need to check
                    var linTerm = lin.Terms[i];
                    var linWeight = lin.Weights[i];

                    if (linTerm is Product nestedProd)
                    {
                        // This is a Product in a LinExpr - we need to expand it with expr2
                        // First process the nested product to get its components
                        var tempLinTerms = new List<Expr>();
                        var tempLinWeights = new List<double>();
                        var tempQuadTerms1 = new List<Expr>();
                        var tempQuadTerms2 = new List<Expr>();
                        var tempQuadWeights = new List<double>();
                        var tempConst = 0.0;

                        ProcessProduct(nestedProd, linWeight, ref tempConst, tempLinTerms, tempLinWeights, tempQuadTerms1, tempQuadTerms2, tempQuadWeights);

                        // Now multiply each result by expr2
                        for (int k = 0; k < tempLinTerms.Count; k++)
                        {
                            quadTerms1.Add(tempLinTerms[k]);
                            quadTerms2.Add(expr2);
                            quadWeights.Add(productWeight * tempLinWeights[k]);
                        }
                        // Quadratic terms from the nested product can't be multiplied by expr2
                        // (that would be cubic), so just add them scaled
                        for (int k = 0; k < tempQuadTerms1.Count; k++)
                        {
                            // This creates a higher-order term - can't represent in QuadExpr
                            // Add back as a Product
                            var higherOrder = new Product([new Constant(tempQuadWeights[k]), tempQuadTerms1[k], tempQuadTerms2[k], expr2]);
                            linearTerms.Add(higherOrder);
                            linearWeights.Add(productWeight);
                        }
                    }
                    else
                    {
                        // Normal case: linear term * expr2
                        quadTerms1.Add(linTerm);
                        quadTerms2.Add(expr2);
                        quadWeights.Add(productWeight * linWeight);
                    }
                }
            }
            // Expand other * LinExpr
            else if (expr2 is LinExpr linRight)
            {
                if (linRight.ConstantTerm != 0.0)
                    ProcessTerm(expr1, productWeight * linRight.ConstantTerm, ref constantSum, linearTerms, linearWeights, quadTerms1, quadTerms2, quadWeights);

                for (int i = 0; i < linRight.Terms.Count; i++)
                {
                    // Same as above - check if the term is a Product
                    var linTerm = linRight.Terms[i];
                    var linWeight = linRight.Weights[i];

                    if (linTerm is Product nestedProd)
                    {
                        // Process the nested product and multiply by expr1
                        var tempLinTerms = new List<Expr>();
                        var tempLinWeights = new List<double>();
                        var tempQuadTerms1 = new List<Expr>();
                        var tempQuadTerms2 = new List<Expr>();
                        var tempQuadWeights = new List<double>();
                        var tempConst = 0.0;

                        ProcessProduct(nestedProd, linWeight, ref tempConst, tempLinTerms, tempLinWeights, tempQuadTerms1, tempQuadTerms2, tempQuadWeights);

                        for (int k = 0; k < tempLinTerms.Count; k++)
                        {
                            quadTerms1.Add(expr1);
                            quadTerms2.Add(tempLinTerms[k]);
                            quadWeights.Add(productWeight * tempLinWeights[k]);
                        }
                        // Higher-order terms
                        for (int k = 0; k < tempQuadTerms1.Count; k++)
                        {
                            linearTerms.Add(new Product([new Constant(tempQuadWeights[k]), expr1, tempQuadTerms1[k], tempQuadTerms2[k]]));
                            linearWeights.Add(productWeight);
                        }
                    }
                    else
                    {
                        quadTerms1.Add(expr1);
                        quadTerms2.Add(linTerm);
                        quadWeights.Add(productWeight * linWeight);
                    }
                }
            }
            else
            {
                // General bilinear: x * y
                quadTerms1.Add(expr1);
                quadTerms2.Add(expr2);
                quadWeights.Add(productWeight);
            }
        }
        else
        {
            // 3+ non-constant factors: this is higher than quadratic
            // Reconstruct the product and add as linear term (will fail IsAtMostQuadratic check later)
            var reconstructed = prod.Factors.Count == prod.Factors.Count
                ? (Expr)prod
                : new Product([new Constant(productWeight / weight), .. prod.Factors]);
            linearTerms.Add(reconstructed);
            linearWeights.Add(weight);
        }
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var result = ConstantTerm;

        // Linear terms
        for (int i = 0; i < LinearTerms.Count; i++)
            result += LinearWeights[i] * LinearTerms[i].Evaluate(x);

        // Quadratic terms
        for (int i = 0; i < QuadraticTerms1.Count; i++)
            result += QuadraticWeights[i] * QuadraticTerms1[i].Evaluate(x) * QuadraticTerms2[i].Evaluate(x);

        return result;
    }

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        // Linear terms
        for (int i = 0; i < LinearTerms.Count; i++)
            LinearTerms[i].AccumulateGradientCompact(x, compactGrad, multiplier * LinearWeights[i], sortedVarIndices);

        // Quadratic terms
        for (int i = 0; i < QuadraticTerms1.Count; i++)
        {
            var f = QuadraticTerms1[i].Evaluate(x);
            var g = QuadraticTerms2[i].Evaluate(x);
            var w = multiplier * QuadraticWeights[i];

            QuadraticTerms1[i].AccumulateGradientCompact(x, compactGrad, w * g, sortedVarIndices);
            QuadraticTerms2[i].AccumulateGradientCompact(x, compactGrad, w * f, sortedVarIndices);
        }
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // Linear terms contribute nothing to Hessian

        // Quadratic terms: ∂²/∂x_i∂x_j (w * f * g)
        // For simple variable products (most common case): x_i * x_j → coefficient w
        for (int i = 0; i < QuadraticTerms1.Count; i++)
        {
            var w = multiplier * QuadraticWeights[i];

            // Simplified for Variable * Variable case (most common for quadratic expressions)
            if (QuadraticTerms1[i] is Variable v1 && QuadraticTerms2[i] is Variable v2)
            {
                int idx1 = v1.Index;
                int idx2 = v2.Index;

                if (idx1 == idx2)
                {
                    // Diagonal: x^2 → d²/dx² = 2
                    hess.Add(idx1, idx1, 2.0 * w);
                }
                else
                {
                    // Off-diagonal: x*y → d²/dxdy = 1 (coefficient stored once, used for both i,j and j,i)
                    hess.Add(idx1, idx2, w);
                }
            }
            else
            {
                // For complex expressions, delegate to their Hessian computation
                // This handles cases like (expr1 * expr2) where expr1/expr2 aren't simple variables
                QuadraticTerms1[i].AccumulateHessian(x, hess, w * QuadraticTerms2[i].Evaluate(x));
                QuadraticTerms2[i].AccumulateHessian(x, hess, w * QuadraticTerms1[i].Evaluate(x));
            }
        }
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        foreach (var term in LinearTerms)
            term.CollectVariables(variables);
        foreach (var term in QuadraticTerms1)
            term.CollectVariables(variables);
        foreach (var term in QuadraticTerms2)
            term.CollectVariables(variables);
    }

    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        foreach (var term in LinearTerms)
            term.CollectHessianSparsity(entries);

        // Quadratic terms contribute to Hessian sparsity
        for (int i = 0; i < QuadraticTerms1.Count; i++)
        {
            if (QuadraticTerms1[i] is Variable v1 && QuadraticTerms2[i] is Variable v2)
                AddSparsityEntry(entries, v1.Index, v2.Index);
            else
            {
                QuadraticTerms1[i].CollectHessianSparsity(entries);
                QuadraticTerms2[i].CollectHessianSparsity(entries);
            }
        }
    }

    protected override bool IsConstantWrtXCore() =>
        LinearTerms.All(t => t.IsConstantWrtX()) &&
        QuadraticTerms1.All(t => t.IsConstantWrtX()) &&
        QuadraticTerms2.All(t => t.IsConstantWrtX());

    protected override bool IsLinearCore() => QuadraticTerms1.Count == 0 && LinearTerms.All(t => t.IsLinear());
    protected override bool IsAtMostQuadraticCore() =>
        LinearTerms.All(t => t.IsAtMostQuadratic()) &&
        QuadraticTerms1.All(t => t.IsLinear()) &&
        QuadraticTerms2.All(t => t.IsLinear());

    protected override Expr CloneCore()
    {
        return new QuadExpr()
        {
            LinearTerms = [.. LinearTerms],
            LinearWeights = [.. LinearWeights],
            QuadraticTerms1 = [.. QuadraticTerms1],
            QuadraticTerms2 = [.. QuadraticTerms2],
            QuadraticWeights = [.. QuadraticWeights],
            ConstantTerm = ConstantTerm
        };
    }

    protected override void PrepareChildren()
    {
        foreach (var term in LinearTerms)
            term.Prepare();
        foreach (var term in QuadraticTerms1)
            term.Prepare();
        foreach (var term in QuadraticTerms2)
            term.Prepare();
    }

    protected override void ClearChildren()
    {
        foreach (var term in LinearTerms)
            term.Clear();
        foreach (var term in QuadraticTerms1)
            term.Clear();
        foreach (var term in QuadraticTerms2)
            term.Clear();
    }

    protected override string ToStringCore()
    {
        // If all terms are simple (variables/constants), format inline
        if (LinearTerms.All(t => t.IsSimpleForPrinting()) &&
               QuadraticTerms1.All(t => t.IsSimpleForPrinting()) &&
               QuadraticTerms2.All(t => t.IsSimpleForPrinting()))
        {
            var result = new StringBuilder();
            result.Append("QuadExpr: ");
            result.Append(ConstantTerm.ToString());

            for (int i = 0; i < LinearTerms.Count; i++)
            {
                var weight = LinearWeights[i];
                var termStr = LinearTerms[i].ToString();

                if (weight == 1)
                    result.Append($" + {termStr}");
                else if (weight == -1)
                    result.Append($" - {termStr}");
                else if (weight >= 0)
                    result.Append($" + {weight}*{termStr}");
                else
                    result.Append($" - {-weight}*{termStr}");
            }

            for (int i = 0; i < QuadraticTerms1.Count; i++)
            {
                var weight = QuadraticWeights[i];
                var term1Str = QuadraticTerms1[i].ToString();
                var term2Str = QuadraticTerms2[i].ToString();
                var quadStr = $"{term1Str}*{term2Str}";

                if (weight == 1)
                    result.Append($" + {quadStr}");
                else if (weight == -1)
                    result.Append($" - {quadStr}");
                else if (weight >= 0)
                    result.Append($" + {weight}*{quadStr}");
                else
                    result.Append($" - {-weight}*{quadStr}");
            }

            return result.ToString();
        }

        // Otherwise, use multi-line tree format
        var sb = new StringBuilder();
        sb.AppendLine($"QuadExpr: {LinearTerms.Count} linear, {QuadraticTerms1.Count} quadratic, constant={ConstantTerm}");

        for (int i = 0; i < LinearTerms.Count; i++)
        {
            var termLines = LinearTerms[i].ToString().Split(Environment.NewLine);
            if (termLines.Length == 1)
                sb.AppendLine($"  Linear [{i}] weight={LinearWeights[i]}: {termLines[0]}");
            else
            {
                sb.AppendLine($"  Linear [{i}] weight={LinearWeights[i]}:");
                foreach (var line in termLines)
                    sb.AppendLine($"    {line}");
            }
        }

        for (int i = 0; i < QuadraticTerms1.Count; i++)
        {
            var term1Lines = QuadraticTerms1[i].ToString().Split(Environment.NewLine);
            var term2Lines = QuadraticTerms2[i].ToString().Split(Environment.NewLine);

            sb.AppendLine($"  Quadratic [{i}] weight={QuadraticWeights[i]}:");
            if (term1Lines.Length == 1)
                sb.AppendLine($"    Term1: {term1Lines[0]}");
            else
            {
                sb.AppendLine("    Term1:");
                foreach (var line in term1Lines)
                    sb.AppendLine($"      {line}");
            }

            if (term2Lines.Length == 1)
                sb.AppendLine($"    Term2: {term2Lines[0]}");
            else
            {
                sb.AppendLine("    Term2:");
                foreach (var line in term2Lines)
                    sb.AppendLine($"      {line}");
            }
        }

        return sb.ToString().TrimEnd();
    }
}
