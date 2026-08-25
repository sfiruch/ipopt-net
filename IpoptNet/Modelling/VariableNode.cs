namespace IpoptNet.Modelling;

internal sealed class VariableNode : ExprNode
{
    public Variable Variable { get; }

    public VariableNode(Variable variable)
    {
        Variable = variable;
    }

    internal override double EvaluateCore(ReadOnlySpan<double> x) => x[Variable.Index] * Variable.Scale;

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        // Every variable differentiates as itself here, eliminated or not: a model with implicit
        // blocks is walked in raw mode and reduced afterwards by ReducedDerivatives, which is where
        // an eliminated variable's dependence on the decision vector is accounted for. The Scale
        // factor applies either way, since this node's value is scratch[Index]·Scale.
        compactGrad[Array.BinarySearch(sortedVarIndices, Variable.Index)] += multiplier * Variable.Scale;
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // A variable has no second derivative of its own. For an eliminated one, ∂²v*/∂p∂p is not
        // propagated here at all — ReducedDerivatives obtains it from the adjoint instead.
    }

    internal override void CollectVariables(HashSet<Variable> variables)
    {
        if (Variable.Block is { } block && !block.Model.IsRawMode)
        {
            block.CollectInputVariables(variables);
            return;
        }
        variables.Add(Variable);
    }

    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries)
    {
        if (Variable.Block is { } block && !block.Model.IsRawMode)
        {
            // ∂²v*_j/∂x_dec_k∂x_dec_p can be non-zero for any pair of decision-vector inputs of the
            // block, so the Hessian sparsity contribution is the clique among those inputs. Use the
            // block's cached input list directly to avoid a per-call HashSet<Variable> allocation.
            block.AddInputCliqueToHessianSparsity(entries);
            return;
        }
    }

    internal override bool IsConstantWrtX() => false;
    internal override bool IsLinear() => Variable.Block is null;
    internal override bool IsAtMostQuadratic() => Variable.Block is null;

    public override string ToString() => Variable.Scale != 1.0 ? $"x[{Variable.Index}]*{Variable.Scale}" : $"x[{Variable.Index}]";

    internal override bool IsSimpleForPrinting() => Variable.Block is null;
}
