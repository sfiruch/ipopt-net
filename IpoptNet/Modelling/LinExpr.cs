using System.Text;

namespace IpoptNet.Modelling;

internal sealed class LinExprNode : ExprNode
{
    public List<ExprNode> Terms { get; set; }
    public List<double> Weights { get; set; }
    public double ConstantTerm { get; set; }

    // Prepared fast-path state (see PrepareChildren). Parallel to Terms:
    private int[]? _termVarIndices;      // Variable.Index for plain-variable terms, -1 otherwise
    private double[]? _termScales;       // Variable.Scale for plain-variable terms
    private int[]? _termPositions;       // compactGrad position per plain-variable term
    private int[]? _termPositionsOwner;  // sortedVarIndices instance _termPositions was built for
    private ExprNode[]? _nonlinearTerms; // terms with a (potentially) non-zero Hessian
    private double[]? _nonlinearWeights;

    public LinExprNode()
    {
        Terms = [];
        Weights = [];
    }

    /// <summary>
    /// Efficiently adds a term to this LinExpr with proper weight extraction.
    /// Used by += operator for O(1) appending instead of O(n) copying.
    /// </summary>
    public void AddTerm(ExprNode term, double weight = 1.0)
    {
        var constantSum = ConstantTerm;
        ProcessTerm(term, weight, ref constantSum, Terms, Weights);
        ConstantTerm = constantSum;
    }

    public LinExprNode(List<ExprNode> terms)
    {
        var constantSum = 0.0;
        var nonConstantTerms = new List<ExprNode>(terms.Count);
        var weights = new List<double>(terms.Count);

        foreach (var term in terms)
        {
            if (term is ConstantNode c)
            {
                constantSum += c.Value;
            }
            // Extract weight from Negation
            else if (term is NegationNode neg)
            {
                // Recursively process the negated operand with weight -1
                ProcessTerm(neg.Operand, -1.0, ref constantSum, nonConstantTerms, weights);
            }
            // Merge nested LinExpr
            else if (term is LinExprNode lin)
            {
                constantSum += lin.ConstantTerm;
                for (int i = 0; i < lin.Terms.Count; i++)
                {
                    nonConstantTerms.Add(lin.Terms[i]);
                    weights.Add(lin.Weights[i]);
                }
            }
            // Extract weight from Product with Factor field
            else if (term is ProductNode prod)
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
                    nonConstantTerms.Add(term);
                    weights.Add(1.0);
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

    private static void ProcessTerm(ExprNode term, double weight, ref double constantSum, List<ExprNode> nonConstantTerms, List<double> weights)
    {
        // Skip terms with zero weight
        if (weight == 0)
            return;

        // Handle nested structures
        if (term is ConstantNode c)
            constantSum += weight * c.Value;
        else if (term is NegationNode neg)
        {
            // Negate weight and continue
            ProcessTerm(neg.Operand, -weight, ref constantSum, nonConstantTerms, weights);
        }
        else if (term is LinExprNode lin)
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
        else if (term is ProductNode prod)
        {
            // Product now extracts Constants into Factor field
            if (prod.Factors.Count == 1)
                ProcessTerm(prod.Factors[0], weight * prod.Factor, ref constantSum, nonConstantTerms, weights);
            else if (prod.Factors.Count == 0)
                constantSum += weight * prod.Factor;
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

    internal override double EvaluateCore(ReadOnlySpan<double> x)
    {
        var result = ConstantTerm;
        for (int i = 0; i < Terms.Count; i++)
            result += Weights[i] * Terms[i].Evaluate(x);
        return result;
    }

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        var varIndices = _termVarIndices;
        if (varIndices is null)
        {
            // Not prepared — plain dispatch.
            for (int i = 0; i < Terms.Count; i++)
                Terms[i].AccumulateGradientCompact(x, compactGrad, multiplier * Weights[i], sortedVarIndices);
            return;
        }

        // Plain-variable terms write compactGrad[BinarySearch(sortedVarIndices, Index)] += w·Scale.
        // The positions only depend on sortedVarIndices, which is a fixed array instance per call
        // site — memoize them keyed on that instance to skip the per-term binary search.
        if (!ReferenceEquals(_termPositionsOwner, sortedVarIndices))
        {
            for (int i = 0; i < varIndices.Length; i++)
                if (varIndices[i] >= 0)
                    _termPositions![i] = Array.BinarySearch(sortedVarIndices, varIndices[i]);
            _termPositionsOwner = sortedVarIndices;
        }

        var positions = _termPositions!;
        for (int i = 0; i < varIndices.Length; i++)
        {
            if (varIndices[i] >= 0)
                compactGrad[positions[i]] += multiplier * Weights[i] * _termScales![i];
            else
                Terms[i].AccumulateGradientCompact(x, compactGrad, multiplier * Weights[i], sortedVarIndices);
        }
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // Linear terms contribute nothing to the Hessian; iterate only the nonlinear subset
        // (computed at Prepare). Block-eliminated variables count as nonlinear and stay included.
        if (_nonlinearTerms is { } nonlinear)
        {
            for (int i = 0; i < nonlinear.Length; i++)
                nonlinear[i].AccumulateHessian(x, hess, multiplier * _nonlinearWeights![i]);
            return;
        }
        for (int i = 0; i < Terms.Count; i++)
            Terms[i].AccumulateHessian(x, hess, multiplier * Weights[i]);
    }

    internal override void CollectVariables(HashSet<Variable> variables)
    {
        foreach (var term in Terms)
            term.CollectVariables(variables);
    }

    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries)
    {
        foreach (var term in Terms)
            term.CollectHessianSparsity(entries);
    }

    internal override bool IsConstantWrtX() => Terms.All(t => t.IsConstantWrtX());
    internal override bool IsLinear() => Terms.All(t => t.IsLinear());
    internal override bool IsAtMostQuadratic() => Terms.All(t => t.IsAtMostQuadratic());

    internal override void PrepareChildren()
    {
        foreach (var term in Terms)
            term.Prepare(_model);

        // Gradient fast path: for terms that are plain (non-eliminated) variables, record the
        // variable index and scale so AccumulateGradientCompact can write directly instead of
        // dispatching. Eliminated variables keep the virtual path (their behavior is mode-dependent).
        _termVarIndices = new int[Terms.Count];
        _termScales = new double[Terms.Count];
        _termPositions = new int[Terms.Count];
        _termPositionsOwner = null;
        for (int i = 0; i < Terms.Count; i++)
        {
            if (Terms[i] is VariableNode { Variable: { Block: null } v })
            {
                _termVarIndices[i] = v.Index;
                _termScales[i] = v.Scale;
            }
            else
                _termVarIndices[i] = -1;
        }

        // Hessian fast path: only nonlinear terms can contribute.
        var nonlinearCount = 0;
        for (int i = 0; i < Terms.Count; i++)
            if (!Terms[i].IsLinear())
                nonlinearCount++;
        _nonlinearTerms = new ExprNode[nonlinearCount];
        _nonlinearWeights = new double[nonlinearCount];
        var k = 0;
        for (int i = 0; i < Terms.Count; i++)
            if (!Terms[i].IsLinear())
            {
                _nonlinearTerms[k] = Terms[i];
                _nonlinearWeights[k++] = Weights[i];
            }
    }

    internal override void ClearChildren()
    {
        foreach (var term in Terms)
            term.Clear();
        _termVarIndices = null;
        _termScales = null;
        _termPositions = null;
        _termPositionsOwner = null;
        _nonlinearTerms = null;
        _nonlinearWeights = null;
    }

    public override string ToString()
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
            var termLines = Terms[i].ToString()!.Split(Environment.NewLine);
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
