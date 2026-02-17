using System.Text;

namespace IpoptNet.Modelling;

public sealed class LinExpr : Expr
{
    public List<Expr> Terms { get; set; }
    public List<double> Weights { get; set; }
    public double ConstantTerm { get; set; }

    public LinExpr()
    {
        Terms = [];
        Weights = [];
    }

    /// <summary>
    /// Efficiently adds a term to this LinExpr with proper weight extraction.
    /// Used by += operator for O(1) appending instead of O(n) copying.
    /// </summary>
    public void AddTerm(Expr term, double weight = 1.0)
    {
        var constantSum = ConstantTerm;
        ProcessTerm(term, weight, ref constantSum, Terms, Weights);
        ConstantTerm = constantSum;
    }

    public LinExpr(List<Expr> terms)
    {
        var constantSum = 0.0;
        var nonConstantTerms = new List<Expr>(terms.Count);
        var weights = new List<double>(terms.Count);

        foreach (var term in terms)
        {
            // Follow replacement to get actual expression
            var actualTerm = term.GetActual();

            if (actualTerm is Constant c)
            {
                constantSum += c.Value;
            }
            // Extract weight from Negation
            else if (actualTerm is Negation neg)
            {
                // Recursively process the negated operand with weight -1
                ProcessTerm(neg.Operand, -1.0, ref constantSum, nonConstantTerms, weights);
            }
            // Merge nested LinExpr
            else if (actualTerm is LinExpr lin)
            {
                constantSum += lin.ConstantTerm;
                for (int i = 0; i < lin.Terms.Count; i++)
                {
                    nonConstantTerms.Add(lin.Terms[i]);
                    weights.Add(lin.Weights[i]);
                }
            }
            // Extract weight from Product with Factor field
            else if (actualTerm is Product prod)
            {
                // Product now extracts Constants into Factor field
                // If it has exactly 1 factor, we can flatten it
                if (prod.Factors.Count == 1)
                {
                    ProcessTerm(prod.Factors[0], prod.Factor, ref constantSum, nonConstantTerms, weights);
                }
                // If it's a constant product (no factors), add to constant term
                else if (prod.Factors.Count == 0)
                {
                    constantSum += prod.Factor;
                }
                else
                {
                    nonConstantTerms.Add(actualTerm);
                    weights.Add(1.0);
                }
            }
            else
            {
                nonConstantTerms.Add(actualTerm);
                weights.Add(1.0);
            }
        }

        Terms = nonConstantTerms;
        Weights = weights;
        ConstantTerm = constantSum;
    }

    private static void ProcessTerm(Expr term, double weight, ref double constantSum, List<Expr> nonConstantTerms, List<double> weights)
    {
        // Skip terms with zero weight
        if (weight == 0)
            return;

        // Follow replacement to get actual expression
        var actualTerm = term.GetActual();

        // Handle nested structures
        if (actualTerm is Constant c)
            constantSum += weight * c.Value;
        else if (actualTerm is Negation neg)
        {
            // Negate weight and continue
            ProcessTerm(neg.Operand, -weight, ref constantSum, nonConstantTerms, weights);
        }
        else if (actualTerm is LinExpr lin)
        {
            // Merge LinExpr: add all its terms with weights scaled by our weight
            constantSum += weight * lin.ConstantTerm;
            for (int i = 0; i < lin.Terms.Count; i++)
            {
                var scaledWeight = weight * lin.Weights[i];
                nonConstantTerms.Add(lin.Terms[i]);
                weights.Add(scaledWeight);
            }
        }
        else if (actualTerm is Product prod)
        {
            // Product now extracts Constants into Factor field
            if (prod.Factors.Count == 1)
                ProcessTerm(prod.Factors[0], weight * prod.Factor, ref constantSum, nonConstantTerms, weights);
            else if (prod.Factors.Count == 0)
                constantSum += weight * prod.Factor;
            else
            {
                // Multiple factors - keep as Product
                nonConstantTerms.Add(actualTerm);
                weights.Add(weight);
            }
        }
        else
        {
            nonConstantTerms.Add(actualTerm);
            weights.Add(weight);
        }
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var result = ConstantTerm;
        for (int i = 0; i < Terms.Count; i++)
            result += Weights[i] * Terms[i].Evaluate(x);
        return result;
    }

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        for (int i = 0; i < Terms.Count; i++)
            Terms[i].AccumulateGradientCompact(x, compactGrad, multiplier * Weights[i], sortedVarIndices);
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        for (int i = 0; i < Terms.Count; i++)
            Terms[i].AccumulateHessian(x, hess, multiplier * Weights[i]);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        foreach (var term in Terms)
            term.CollectVariables(variables);
    }

    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        foreach (var term in Terms)
            term.CollectHessianSparsity(entries);
    }

    protected override bool IsConstantWrtXCore() => Terms.All(t => t.IsConstantWrtX());
    protected override bool IsLinearCore() => Terms.All(t => t.IsLinear());
    protected override bool IsAtMostQuadraticCore() => Terms.All(t => t.IsAtMostQuadratic());

    protected override Expr CloneCore()
    {
        return new LinExpr([.. Terms])
        {
            Weights = [.. Weights],
            ConstantTerm = ConstantTerm
        };
    }

    protected override void PrepareChildren()
    {
        foreach (var term in Terms)
            term.Prepare();
    }

    protected override void ClearChildren()
    {
        foreach (var term in Terms)
            term.Clear();
    }

    protected override string ToStringCore()
    {
        // If all terms are simple (variables/constants), format inline
        if (Terms.All(t => t.IsSimpleForPrinting()))
        {
            var result = new StringBuilder();
            result.Append("LinExpr: ");
            result.Append(ConstantTerm.ToString());

            for (int i = 0; i < Terms.Count; i++)
            {
                var weight = Weights[i];
                var termStr = Terms[i].ToString();

                if (weight == 1)
                    result.Append($" + {termStr}");
                else if (weight == -1)
                    result.Append($" - {termStr}");
                else if (weight > 0)
                    result.Append($" + {weight}*{termStr}");
                else
                    result.Append($" - {-weight}*{termStr}");
            }

            return result.ToString();
        }

        // Otherwise, use multi-line tree format
        var sb = new StringBuilder();
        sb.AppendLine($"LinExpr: {Terms.Count} terms, constant={ConstantTerm}");
        for (int i = 0; i < Terms.Count; i++)
        {
            var termLines = Terms[i].ToString().Split(Environment.NewLine);
            if (termLines.Length == 1)
                sb.AppendLine($"  [{i}] weight={Weights[i]}: {termLines[0]}");
            else
            {
                sb.AppendLine($"  [{i}] weight={Weights[i]}:");
                foreach (var line in termLines)
                    sb.AppendLine($"    {line}");
            }
        }
        return sb.ToString().TrimEnd();
    }
}
