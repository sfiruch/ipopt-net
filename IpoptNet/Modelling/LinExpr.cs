namespace IpoptNet.Modelling;

public class LinExpr : Expr
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
            if (term is Constant c)
            {
                constantSum += c.Value;
            }
            // Extract weight from Negation
            else if (term is Negation neg)
            {
                // Recursively process the negated operand with weight -1
                ProcessTerm(neg.Operand, -1.0, ref constantSum, nonConstantTerms, weights);
            }
            // Merge nested LinExpr
            else if (term is LinExpr lin)
            {
                constantSum += lin.ConstantTerm;
                for (int i = 0; i < lin.Terms.Count; i++)
                {
                    nonConstantTerms.Add(lin.Terms[i]);
                    weights.Add(lin.Weights[i]);
                }
            }
            // Extract weight from Product with Factor field
            else if (term is Product prod)
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
                    // Multiple factors - keep as Product but account for Factor
                    if (Math.Abs(prod.Factor - 1.0) < 1e-15)
                    {
                        nonConstantTerms.Add(term);
                        weights.Add(1.0);
                    }
                    else
                    {
                        // Need to keep the Product with its Factor
                        nonConstantTerms.Add(term);
                        weights.Add(1.0);
                    }
                }
            }
            else
            {
                nonConstantTerms.Add(term);
                weights.Add(1.0);
            }
        }

        Terms = nonConstantTerms;
        Weights = weights;
        ConstantTerm = constantSum;
    }

    private static void ProcessTerm(Expr term, double weight, ref double constantSum, List<Expr> nonConstantTerms, List<double> weights)
    {
        // Handle nested structures
        if (term is Constant c)
        {
            constantSum += weight * c.Value;
        }
        else if (term is Negation neg)
        {
            // Negate weight and continue
            ProcessTerm(neg.Operand, -weight, ref constantSum, nonConstantTerms, weights);
        }
        else if (term is LinExpr lin)
        {
            // Merge LinExpr: add all its terms with weights scaled by our weight
            constantSum += weight * lin.ConstantTerm;
            for (int i = 0; i < lin.Terms.Count; i++)
            {
                nonConstantTerms.Add(lin.Terms[i]);
                weights.Add(weight * lin.Weights[i]);
            }
        }
        else if (term is Product prod)
        {
            // Product now extracts Constants into Factor field
            if (prod.Factors.Count == 1)
            {
                ProcessTerm(prod.Factors[0], weight * prod.Factor, ref constantSum, nonConstantTerms, weights);
            }
            else if (prod.Factors.Count == 0)
            {
                constantSum += weight * prod.Factor;
            }
            else
            {
                // Multiple factors - keep as Product
                nonConstantTerms.Add(term);
                weights.Add(weight);
            }
        }
        else
        {
            nonConstantTerms.Add(term);
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

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, Dictionary<int, int> varIndexToCompact)
    {
        for (int i = 0; i < Terms.Count; i++)
            Terms[i].AccumulateGradientCompact(x, compactGrad, multiplier * Weights[i], varIndexToCompact);
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
        var clone = new LinExpr([.. Terms]);
        clone.Weights = [.. Weights];
        clone.ConstantTerm = ConstantTerm;
        return clone;
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

    protected override void PrintCore(TextWriter writer, string indent)
    {
        writer.WriteLine($"{indent}LinExpr: {Terms.Count} terms, constant={ConstantTerm}");
        for (int i = 0; i < Terms.Count; i++)
        {
            writer.WriteLine($"{indent}  [{i}] weight={Weights[i]}:");
            Terms[i].Print(writer, indent + "    ");
        }
    }
}
