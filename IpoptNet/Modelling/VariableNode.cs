namespace IpoptNet.Modelling;

internal sealed class VariableNode : ExprNode
{
    public Variable Variable { get; }

    public VariableNode(Variable variable)
    {
        Variable = variable;
    }

    internal override double Evaluate(ReadOnlySpan<double> x) => x[Variable.Index] * Variable.Scale;

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        // Eliminated variables in redirect mode propagate the seed through their block's
        // implicit-function-theorem chain. In raw mode (during a block's own Solve / PropagateGradient),
        // they behave as plain variables — write to compactGrad at the variable's index.
        if (Variable.Block is { } block && !ImplicitBlock.IsRawMode)
        {
            block.PropagateGradient(Variable.IndexInBlock, x, compactGrad, multiplier, sortedVarIndices);
            return;
        }
        compactGrad[Array.BinarySearch(sortedVarIndices, Variable.Index)] += multiplier * Variable.Scale;
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        if (Variable.Block is { } block && !ImplicitBlock.IsRawMode)
        {
            // ∂²v*_j/∂x_dec_k∂x_dec_p is non-zero in general; let the block propagate.
            block.PropagateHessian(Variable.IndexInBlock, x, hess, multiplier);
            return;
        }
        // Plain variable has no second derivative contribution.
    }

    internal override void CollectVariables(HashSet<Variable> variables)
    {
        if (Variable.Block is { } block && !ImplicitBlock.IsRawMode)
        {
            block.CollectInputVariables(variables);
            return;
        }
        variables.Add(Variable);
    }

    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries)
    {
        if (Variable.Block is { } block && !ImplicitBlock.IsRawMode)
        {
            // ∂²v*_j/∂x_dec_k∂x_dec_p can be non-zero for any pair of decision-vector inputs of the
            // block, so the Hessian sparsity contribution is the clique among those inputs.
            var inputs = new HashSet<Variable>();
            block.CollectInputVariables(inputs);
            AddClique(entries, inputs);
            return;
        }
    }

    internal override bool IsConstantWrtX() => false;
    internal override bool IsLinear() => Variable.Block is null;
    internal override bool IsAtMostQuadratic() => Variable.Block is null;

    public override string ToString() => Variable.Scale != 1.0 ? $"x[{Variable.Index}]*{Variable.Scale}" : $"x[{Variable.Index}]";

    internal override bool IsSimpleForPrinting() => Variable.Block is null;
}
