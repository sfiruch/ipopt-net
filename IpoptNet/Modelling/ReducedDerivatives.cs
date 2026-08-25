using System.Numerics.Tensors;

namespace IpoptNet.Modelling;

/// <summary>
/// Computes the reduced gradient, reduced Jacobian rows and reduced Lagrangian Hessian of a model
/// whose state variables are eliminated by <see cref="ImplicitBlock"/>s — the derivatives IPOPT
/// asks for, in the space of the variables it actually sees.
///
/// <para>The blocks define <c>v</c> implicitly through residuals <c>E(p, v) = 0</c> that are linear
/// in their own eliminated variables. Writing <c>X = ∂v/∂p</c>, the outer objective <c>F</c> and
/// constraints <c>g</c> reduce to</para>
/// <code>
///   ∇F_red   = F_p + Xᵀ F_v
///   ∇g_i,red = g_i,p + Xᵀ g_i,v
///   ∇²L_red  = [I; X]ᵀ · ( ∇²L + Σ_l μ_l ∇²E_l ) · [I; X]
/// </code>
/// <para>with <c>L = σF + Σ λ_i g_i</c> and the adjoint <c>μ</c> solving <c>A_vvᵀ μ = −∂L/∂v</c>.
/// Everything on the right-hand sides is a *local* derivative of an expression as written, taken in
/// raw mode where an eliminated variable behaves as an ordinary one.</para>
///
/// <para>Both linear systems are solved by substitution rather than by any general sparse
/// factorization: <see cref="Model.AddImplicitBlock"/> requires blocks in topological order, so a
/// block's residuals may only reference *earlier* blocks' eliminated variables and <c>A_vv</c> is
/// block lower triangular. Forward substitution gives <c>X</c>, backward substitution over the same
/// factors gives <c>μ</c>, and each diagonal block reuses the LU the block already computed for its
/// own forward solve. This holds for any block DAG, not just a chain.</para>
///
/// <para>The alternative — propagating ∂²v/∂p∂p block by block — has to materialise an N×N tensor
/// per eliminated state, so its cost grows with the square of the input closure and it walks a long
/// dependent chain of small updates. The adjoint form replaces that with one factorization reused
/// across all N columns plus dense linear algebra.</para>
/// </summary>
internal sealed class ReducedDerivatives
{
    private readonly Model _model;
    private readonly ImplicitBlock[] _blocks;
    private readonly double[] _scratch;

    // p-space: the variables IPOPT sees. Column c ↔ _pIndices[c] ↔ compact index c.
    private readonly int[] _pIndices;
    private readonly int[] _pColumnOf;      // Variable.Index → column, -1 if not a p variable
    private readonly bool[] _frozenColumn;  // pinned by equal bounds: derivatives w.r.t. it are not asked for
    private readonly int _n;

    // v-space: every variable eliminated by this plan's blocks, grouped by block in topological order.
    private readonly int[] _vRowOf;        // Variable.Index → row, -1 if not eliminated here
    private readonly int[] _blockRowOffset;
    private readonly int _nv;

    // X, stored column-major (one contiguous row of length _nv per p column) because the dominant
    // cost is the final N²/2 dot products over v-space, which want contiguous columns.
    private readonly double[] _x;          // _n × _nv
    private readonly double[] _y;          // _n × _nv, scratch for W_vv · X
    private readonly double[] _blockRhs;   // _n × maxBlockSize, per-block right-hand sides
    private readonly double[] _adjoint;    // _nv
    private readonly double[] _rawGrad;    // totalVars, raw-mode gradient accumulation
    private readonly double[] _reducedRow; // _n, one reduced gradient / Jacobian row
    private readonly double[] _hessBlock;  // _n × _n reduced Hessian, lower triangle used

    // Coupling entries of A_vv below the diagonal, and of A_vp, rebuilt every pass: a block's
    // residuals are only linear in their OWN eliminated variables, so both depend on v* and must be
    // re-read after the forward solves rather than reused from the block's own extraction at v = 0.
    private readonly List<Coupling>[] _coupling;   // per block
    private readonly List<PColumn>[] _pColumns;    // per block

    private readonly Expr[] _outerExprs;           // objective followed by the constraints
    private readonly SparseHessianAccumulator _rawHessian;
    private readonly int[] _rawHessianRows, _rawHessianCols;

    private long _builtGeneration = -1;

    private readonly record struct Coupling(int Residual, int OtherRow, double Value);
    private readonly record struct PColumn(int Residual, int Column, double Value);

    public ReducedDerivatives(Model model, ImplicitBlock[] blocks, Variable[] activeVariables,
        Expr objective, Constraint[] constraints, double[] scratch)
    {
        _model = model;
        _blocks = blocks;
        _scratch = scratch;
        int totalVars = scratch.Length;

        _n = activeVariables.Length;
        _pIndices = new int[_n];
        _pColumnOf = new int[totalVars];
        _frozenColumn = new bool[_n];
        Array.Fill(_pColumnOf, -1);

        // Same rule ImplicitBlock.IsConstantInput applies to its closure: a variable pinned by equal
        // bounds holds one value for the whole solve, so no derivative with respect to it is ever
        // asked for and the redirect-mode sparsity never declared its entries. Leaving its column of
        // X at zero keeps the reduced derivatives matching that structure exactly. The exception is
        // IPOPT's derivative checker, which does ask about a fixed variable's entries — while it is
        // on, the optimization stands down here too.
        bool treatFixedAsConstant =
            (model.Options.FixedVariableTreatment is null or FixedVariableTreatment.MakeParameter)
            && model.Options.DerivativeTest is null or DerivativeTest.None;

        for (int c = 0; c < _n; c++)
        {
            var variable = activeVariables[c];
            _pIndices[c] = variable.Index;
            _pColumnOf[variable.Index] = c;
            _frozenColumn[c] = treatFixedAsConstant
                && variable.LowerBound == variable.UpperBound && double.IsFinite(variable.LowerBound);
        }

        _vRowOf = new int[totalVars];
        Array.Fill(_vRowOf, -1);
        _blockRowOffset = new int[blocks.Length + 1];
        int maxBlockSize = 1;
        for (int i = 0; i < blocks.Length; i++)
        {
            _blockRowOffset[i] = _nv;
            foreach (var v in blocks[i].Variables)
                _vRowOf[v.Index] = _nv++;
            maxBlockSize = Math.Max(maxBlockSize, blocks[i].Variables.Length);
        }
        _blockRowOffset[blocks.Length] = _nv;

        long sensitivityEntries = (long)_n * _nv;
        if (sensitivityEntries > int.MaxValue / 2)
            throw new InvalidOperationException(
                $"Reduced derivatives need two {_n} × {_nv} sensitivity matrices, which exceed the maximum array size. " +
                "Reduce the fit window, the number of eliminated states, or the number of free parameters.");
        _x = new double[sensitivityEntries];
        _y = new double[sensitivityEntries];
        _blockRhs = new double[_n * maxBlockSize];
        _adjoint = new double[_nv];
        _rawGrad = new double[totalVars];
        _reducedRow = new double[_n];
        _hessBlock = new double[_n * _n];

        _coupling = new List<Coupling>[blocks.Length];
        _pColumns = new List<PColumn>[blocks.Length];
        for (int i = 0; i < blocks.Length; i++)
        {
            _coupling[i] = [];
            _pColumns[i] = [];
        }

        _outerExprs = new Expr[1 + constraints.Length];
        _outerExprs[0] = objective;
        for (int i = 0; i < constraints.Length; i++)
            _outerExprs[1 + i] = constraints[i].Expression;

        // Raw-mode Hessian structure: the local second derivatives of the outer functions and of
        // every residual, over p ∪ v. Sparse and block-local — this is what replaces the dense
        // per-state tensors.
        var entries = new HashSet<(int row, int col)>();
        using (model.EnterRawMode())
        {
            foreach (var e in _outerExprs)
                e.CollectHessianSparsity(entries);
            foreach (var block in blocks)
                foreach (var residual in block.Residuals)
                    residual.CollectHessianSparsity(entries);
        }
        // Sorted by (row, col): SparseHessianAccumulator keeps the structure in the order given and
        // binary-searches within each row, so the order is part of its contract — and it lets the
        // contraction below read Values[e] against _rawHessianRows[e] positionally.
        var sorted = entries.OrderBy(e => e.row).ThenBy(e => e.col).ToArray();
        _rawHessianRows = new int[sorted.Length];
        _rawHessianCols = new int[sorted.Length];
        for (int e = 0; e < sorted.Length; e++)
            (_rawHessianRows[e], _rawHessianCols[e]) = sorted[e];
        _rawHessian = new SparseHessianAccumulator(totalVars, _rawHessianRows, _rawHessianCols);
    }

    /// <summary>Number of eliminated states this reduction covers — diagnostics only.</summary>
    public int StateCount => _nv;

    /// <summary>Builds X = ∂v/∂p for the current iterate, once per evaluation pass. The blocks'
    /// forward solves must already have run, so v* and each block's LU are current.</summary>
    private void EnsureBuilt()
    {
        if (_builtGeneration == _model.EvalGeneration) return;
        _builtGeneration = _model.EvalGeneration;

        using var raw = _model.EnterRawMode();
        for (int i = 0; i < _blocks.Length; i++)
        {
            var block = _blocks[i];
            int size = block.Variables.Length;
            int offset = _blockRowOffset[i];
            var coupling = _coupling[i];
            var pColumns = _pColumns[i];
            coupling.Clear();
            pColumns.Clear();

            // Read each residual's raw-mode gradient AT v*. The own-variable columns are A_ii, which
            // the block already factorized (constant in v by the linearity requirement); the rest is
            // the coupling to earlier blocks and to p, both of which can move with v*.
            for (int l = 0; l < size; l++)
            {
                var residual = block.Residuals[l];
                var indices = residual.SortedVariableIndices;
                foreach (var idx in indices)
                    _rawGrad[idx] = 0.0;
                residual.AccumulateGradient(_scratch, _rawGrad);
                foreach (var idx in indices)
                {
                    double value = _rawGrad[idx];
                    if (value == 0.0) continue;
                    int row = _vRowOf[idx];
                    if (row >= offset && row < offset + size) continue;   // own column: in A_ii
                    if (row >= 0)
                        coupling.Add(new Coupling(l, row, value));
                    else if (_pColumnOf[idx] is var col && col >= 0)
                        pColumns.Add(new PColumn(l, col, value));
                }
            }

            // rhs = −A_vp − Σ_{j<i} A_ij · X_j, then X_i = LU_i \ rhs, one column at a time.
            Array.Clear(_blockRhs, 0, _n * size);
            foreach (var p in pColumns)
                if (!_frozenColumn[p.Column])
                    _blockRhs[p.Column * size + p.Residual] -= p.Value;
            foreach (var c in coupling)
                for (int col = 0; col < _n; col++)
                    if (!_frozenColumn[col])
                        _blockRhs[col * size + c.Residual] -= c.Value * _x[col * _nv + c.OtherRow];

            for (int col = 0; col < _n; col++)
            {
                if (_frozenColumn[col]) continue;   // stays zero: nothing differentiates by it
                var rhs = _blockRhs.AsSpan(col * size, size);
                block.SolveWithFactor(rhs);
                rhs.CopyTo(_x.AsSpan(col * _nv + offset, size));
            }
        }
    }

    /// <summary>Reduced gradient of one outer expression: <c>e_p + Xᵀ e_v</c>, written into
    /// <paramref name="compactGrad"/> (IPOPT column space, cleared by the caller).</summary>
    public void Gradient(Expr expr, Span<double> compactGrad)
    {
        EnsureBuilt();
        using var raw = _model.EnterRawMode();
        var indices = expr.SortedVariableIndices;
        foreach (var idx in indices)
            _rawGrad[idx] = 0.0;
        expr.AccumulateGradient(_scratch, _rawGrad);

        foreach (var idx in indices)
        {
            double value = _rawGrad[idx];
            if (value == 0.0) continue;
            int col = _pColumnOf[idx];
            if (col >= 0)
            {
                compactGrad[col] += value;
                continue;
            }
            int row = _vRowOf[idx];
            if (row < 0) continue;   // pinned or outside this partition: constant here
            for (int c = 0; c < _n; c++)
                compactGrad[c] += value * _x[c * _nv + row];
        }
    }

    /// <summary>Reduced Lagrangian Hessian for IPOPT's obj_factor and multipliers, accumulated into
    /// <paramref name="target"/> (original Variable.Index space, p variables only).</summary>
    public void LagrangianHessian(double objFactor, ReadOnlySpan<double> lambda, HessianAccumulator target)
    {
        EnsureBuilt();
        using var raw = _model.EnterRawMode();

        // ∂L/∂v, for the adjoint.
        Array.Clear(_adjoint);
        AccumulateOuterGradientIntoAdjoint(_outerExprs[0], objFactor);
        for (int i = 0; i < lambda.Length; i++)
            if (lambda[i] != 0.0)
                AccumulateOuterGradientIntoAdjoint(_outerExprs[1 + i], lambda[i]);

        // μ solves A_vvᵀ μ = −∂L/∂v. The transpose of a block lower-triangular matrix is block upper
        // triangular, so this is a backward substitution: solve the last block first, then push its
        // contribution up into the earlier rows it couples to.
        for (int r = 0; r < _nv; r++)
            _adjoint[r] = -_adjoint[r];
        for (int i = _blocks.Length - 1; i >= 0; i--)
        {
            var block = _blocks[i];
            int size = block.Variables.Length;
            int offset = _blockRowOffset[i];
            block.SolveTransposeWithFactor(_adjoint.AsSpan(offset, size));
            foreach (var c in _coupling[i])
                _adjoint[c.OtherRow] -= c.Value * _adjoint[offset + c.Residual];
        }

        // W = ∇²L + Σ_l μ_l ∇²E_l, all local, all in raw mode.
        _rawHessian.Clear();
        _outerExprs[0].AccumulateHessian(_scratch, _rawHessian, objFactor);
        for (int i = 0; i < lambda.Length; i++)
            if (lambda[i] != 0.0)
                _outerExprs[1 + i].AccumulateHessian(_scratch, _rawHessian, lambda[i]);
        for (int i = 0; i < _blocks.Length; i++)
        {
            var block = _blocks[i];
            int offset = _blockRowOffset[i];
            for (int l = 0; l < block.Variables.Length; l++)
            {
                double mu = _adjoint[offset + l];
                if (mu != 0.0)
                    block.Residuals[l].AccumulateHessian(_scratch, _rawHessian, mu);
            }
        }

        // Contract: H = [I; X]ᵀ W [I; X].
        Array.Clear(_hessBlock, 0, _n * _n);
        Array.Clear(_y, 0, _n * _nv);
        bool anyStateCurvature = false;
        var values = _rawHessian.Values;
        for (int e = 0; e < values.Length; e++)
        {
            double w = values[e];
            if (w == 0.0) continue;
            int i = _rawHessianRows[e], j = _rawHessianCols[e];
            int ci = _pColumnOf[i], cj = _pColumnOf[j];
            int ri = _vRowOf[i], rj = _vRowOf[j];

            if (ci >= 0 && cj >= 0)
            {
                AddSymmetric(ci, cj, w);
            }
            else if (ci >= 0 && rj >= 0)
            {
                AddCross(ci, rj, w);
            }
            else if (ri >= 0 && cj >= 0)
            {
                AddCross(cj, ri, w);
            }
            else if (ri >= 0 && rj >= 0)
            {
                // Deferred into Y so the state-space part becomes one dense product below rather
                // than an N² outer product per entry.
                anyStateCurvature = true;
                for (int c = 0; c < _n; c++)
                    _y[c * _nv + ri] += w * _x[c * _nv + rj];
                if (ri != rj)
                    for (int c = 0; c < _n; c++)
                        _y[c * _nv + rj] += w * _x[c * _nv + ri];
            }
            // Anything else touches a pinned variable or one outside this partition: constant here.
        }

        if (anyStateCurvature)
            for (int c1 = 0; c1 < _n; c1++)
            {
                var xc1 = _x.AsSpan(c1 * _nv, _nv);
                for (int c2 = 0; c2 <= c1; c2++)
                    _hessBlock[c1 * _n + c2] += TensorPrimitives.Dot<double>(xc1, _y.AsSpan(c2 * _nv, _nv));
            }

        for (int c1 = 0; c1 < _n; c1++)
            for (int c2 = 0; c2 <= c1; c2++)
            {
                double value = _hessBlock[c1 * _n + c2];
                if (value != 0.0)
                    target.Add(_pIndices[c1], _pIndices[c2], value);
            }
    }

    /// <summary>Adds the contribution of a W entry pairing parameter <paramref name="column"/> with
    /// state <paramref name="row"/>. Such an entry stands for both (p, v) and (v, p) in the full
    /// symmetric matrix, and both orientations feed the same reduced entry: every off-diagonal
    /// column picks the term up once, but the reduced diagonal at that parameter picks it up twice.
    /// Missing the second one leaves the diagonal short — the exact shape of a derivative-checker
    /// failure that only shows on obj_hess[c, c].</summary>
    private void AddCross(int column, int row, double w)
    {
        for (int c = 0; c < _n; c++)
            AddSymmetric(column, c, w * _x[c * _nv + row]);
        AddSymmetric(column, column, w * _x[column * _nv + row]);
    }

    /// <summary>Lower-triangle accumulation into the reduced Hessian.
    ///
    /// <para>Pairs touching a pinned column are dropped rather than accumulated. A pinned variable
    /// holds one value for the whole solve, so the model is the one where that value was written as
    /// a literal: the closure excludes it, the declared sparsity therefore has no entry for it, and
    /// IPOPT — which takes it out of the problem under MakeParameter — never reads one. Residual
    /// second derivatives are where this bites, since they reach the reduced Hessian only through
    /// the block-input clique that already left the pinned variable out.</para></summary>
    private void AddSymmetric(int c1, int c2, double value)
    {
        if (value == 0.0) return;
        if (_frozenColumn[c1] || _frozenColumn[c2]) return;
        if (c1 < c2) (c1, c2) = (c2, c1);
        _hessBlock[c1 * _n + c2] += value;
    }

    private void AccumulateOuterGradientIntoAdjoint(Expr expr, double weight)
    {
        var indices = expr.SortedVariableIndices;
        foreach (var idx in indices)
            _rawGrad[idx] = 0.0;
        expr.AccumulateGradient(_scratch, _rawGrad, weight);
        foreach (var idx in indices)
        {
            int row = _vRowOf[idx];
            if (row >= 0)
                _adjoint[row] += _rawGrad[idx];
        }
    }
}
