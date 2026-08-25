using System.Buffers;
using System.Numerics.Tensors;

namespace IpoptNet.Modelling;

/// <summary>
/// Represents a set of variables eliminated from the IPOPT decision vector by an implicit
/// linear system A(other)·v = b(other), where v are the "eliminated" variables and "other" are
/// inputs (parameters, decision-vector variables, or other blocks' eliminated variables).
///
/// At each fresh IPOPT evaluation pass, the block solves the small linear system numerically
/// and writes v* into the per-pass scratch buffer (indexed by Variable.Index). Upstream
/// expressions that reference an eliminated variable read v* via VariableNode's redirect path,
/// and gradients propagate through the implicit-function theorem.
///
/// Constraint expressions handed to AddImplicitBlock must be linear in the eliminated variables
/// (they may be arbitrary in the inputs).
/// </summary>
internal sealed class ImplicitBlock
{
    public Model Model { get; }
    public Variable[] Variables { get; }
    public Expr[] Residuals { get; }

    private readonly int _n;
    private readonly double[] _A;        // _n × _n, row-major; A[i,j] = ∂E_i/∂v_j (constant in v by linearity)
    private readonly double[] _b;        // _n; b_i = -E_i evaluated at v=0
    private readonly double[] _vstar;    // _n; A·v* = b
    private readonly int[] _perm;        // LU permutation
    private readonly double[] _luFactor; // LU decomposition (in-place over a copy of _A)
    private long _generation = -1;       // last eval-pass id we solved for

    /// <summary>Sorted array of original Variable.Index values for decision-vector inputs of this
    /// block (transitive across other blocks). Built from _inputs on first need.</summary>
    private int[]? _inputIndicesSorted;

    /// <summary>All variables (own elim + transitive non-elim inputs) referenced by residuals
    /// in raw mode. Used to iterate non-zero gradient entries during PropagateGradient. Cached
    /// after PrepareResiduals.</summary>
    private Variable[]? _residualVarsRaw;

    /// <summary>Variable.Index values of <see cref="_residualVarsRaw"/>, used to selectively
    /// zero just the entries a raw-mode AccumulateGradient call writes into (instead of
    /// Array.Clear-ing the totalVars-sized gradient buffer, which is hugely wasteful).</summary>
    private int[]? _residualVarIndices;

    /// <summary>Inputs reachable from the residual expressions (transitively across other blocks),
    /// excluding this block's own eliminated variables. Cached on first request.</summary>
    private HashSet<Variable>? _inputs;
    private bool _collectingInputs;

    /// <summary>Whether a variable pinned by equal bounds counts as a constant rather than an input
    /// — see <see cref="IsConstantInput"/>. Latched once per solve in <see cref="PrepareResiduals"/>
    /// so every block in a solve applies the same rule (the closures are cross-checked against each
    /// other, and a per-call read of Options could not be relied on to stay put).</summary>
    private bool _treatFixedAsConstant;

    public ImplicitBlock(Model model, Variable[] variables, Expr[] residuals)
    {
        if (variables.Length != residuals.Length)
            throw new ArgumentException("Number of variables must match number of residual expressions.");
        Model = model;
        Variables = variables;
        Residuals = residuals;
        _n = variables.Length;
        _A = new double[_n * _n];
        _b = new double[_n];
        _vstar = new double[_n];
        _perm = new int[_n];
        _luFactor = new double[_n * _n];
    }

    /// <summary>Prepare residual expressions: walk the AST in raw mode so each residual's
    /// _cachedVariables / _sortedVarIndices include its own eliminated vars (needed to read
    /// gradient[v.Index] when extracting A_{i,j}).</summary>
    public void PrepareResiduals()
    {
        // A variable's bounds are public mutable fields, so which of this block's inputs are pinned
        // — and hence the shape of every cache sized by the input count — is only settled once the
        // caller hands the model to Solve. Latch the rule and drop everything derived from the old
        // one; a solve that reuses a model after freeing a pinned variable would otherwise assemble
        // derivatives against a closure that no longer describes it. Silent, not a crash: the
        // generation guards stay monotonic across solves and would not notice.
        _treatFixedAsConstant =
            (Model.Options.FixedVariableTreatment is null or FixedVariableTreatment.MakeParameter)
            // IPOPT's derivative checker asks about a fixed variable's entries even though the solve
            // proper never does — it runs before the variable is taken out of the problem, while the
            // solve is bit-identical to one where the value was a literal constant. Reporting zeros
            // to it would turn a diagnostic into a source of false alarms, so while it is switched
            // on the optimization stands down and pinned inputs keep their columns.
            && Model.Options.DerivativeTest is null or DerivativeTest.None;
        _inputs = null;
        _inputIndicesSorted = null;

        using (Model.EnterRawMode())
            foreach (var r in Residuals)
                r.Prepare(Model);

        // Snapshot the union of residuals' raw-mode cached variables for fast iteration during
        // PropagateGradient. CollectVariables in raw mode adds Variables themselves (not their
        // blocks' inputs), so this includes own elim + other elim + non-elim inputs.
        var union = new HashSet<Variable>();
        using (Model.EnterRawMode())
            foreach (var r in Residuals)
                r.CollectVariables(union);
        _residualVarsRaw = [.. union];
        _residualVarIndices = _residualVarsRaw.Select(v => v.Index).ToArray();
    }

    public void ClearResiduals()
    {
        foreach (var r in Residuals)
            r.Clear();
    }

    /// <summary>A variable pinned by equal bounds contributes no derivative: it holds one value for
    /// the whole solve, so ∂v*/∂x and ∂²v*/∂x² with respect to it are never asked for. Treating it
    /// as a constant keeps it out of this block's input closure, and the closure's size is what the
    /// dense sensitivity vectors and the N² second-order cache are sized by — so a model that pins
    /// its parameters or its initial conditions stops paying for columns that could never move.
    ///
    /// Only sound while IPOPT itself removes the variable from the problem, which is what
    /// <see cref="FixedVariableTreatment.MakeParameter"/> (its default) does. Under MakeConstraint or
    /// RelaxBounds the variable stays in the NLP as a genuine unknown — held in place by a constraint
    /// or by all-but-equal bounds — and IPOPT needs its true derivatives, so the rule is off there.
    ///
    /// Infinite bounds are excluded explicitly: a variable free on both sides with LowerBound ==
    /// UpperBound == ±∞ is degenerate, not pinned, and must not be mistaken for a constant.</summary>
    private bool IsConstantInput(Variable v) =>
        _treatFixedAsConstant && v.Block is null && v.LowerBound == v.UpperBound && double.IsFinite(v.LowerBound);

    /// <summary>Computes the transitive closure of decision-vector variables this block depends on,
    /// for upstream sparsity analysis. Excludes this block's own eliminated variables, and any input
    /// pinned to a constant by equal bounds (see <see cref="IsConstantInput"/>).</summary>
    public void CollectInputVariables(HashSet<Variable> result)
    {
        if (_inputs is null)
        {
            if (_collectingInputs)
                throw new InvalidOperationException(
                    "ImplicitBlock: cycle detected in CollectInputVariables. Implicit blocks must be added in topological order " +
                    "(checked at AddImplicitBlock; this exception means the topological-order check missed a case).");
            _collectingInputs = true;
            try
            {
                _inputs = new HashSet<Variable>();
                // Use the snapshot if available; otherwise walk in raw mode.
                IEnumerable<Variable> raw;
                if (_residualVarsRaw is not null) raw = _residualVarsRaw;
                else
                {
                    var tmp = new HashSet<Variable>();
                    using (Model.EnterRawMode())
                        foreach (var r in Residuals)
                            r.CollectVariables(tmp);
                    raw = tmp;
                }

                foreach (var v in raw)
                {
                    if (v.Block == this) continue;
                    if (v.Block is { } other)
                        other.CollectInputVariables(_inputs);
                    else if (!IsConstantInput(v))
                        _inputs.Add(v);
                }
            }
            finally { _collectingInputs = false; }
        }

        foreach (var v in _inputs!)
            result.Add(v);
    }

    /// <summary>Verifies the residual expressions are affine in the block's own eliminated variables
    /// by computing A at v=0 and at v=1, and asserting they agree. Called once after PrepareResiduals,
    /// before any IPOPT evaluation pass. Self-contained — rents its own scratch and gradient buffers
    /// from the pool, doesn't touch the caller's per-pass state.
    /// Cheap (one extra AccumulateGradient per residual × 2) and fails fast with a precise error.</summary>
    public void VerifyLinearity(int totalVars)
    {
        var atZero = ArrayPool<double>.Shared.Rent(_n * _n);
        var atOne = ArrayPool<double>.Shared.Rent(_n * _n);
        var scratch = ArrayPool<double>.Shared.Rent(totalVars);
        var gradBuf = ArrayPool<double>.Shared.Rent(totalVars);
        Array.Clear(scratch, 0, totalVars);

        using (Model.EnterRawMode())
        {
            var scratchSpan = new ReadOnlySpan<double>(scratch);
            ExtractA(atZero, scratchSpan, gradBuf, totalVars, ownVarValue: 0.0, scratch);
            ExtractA(atOne, scratchSpan, gradBuf, totalVars, ownVarValue: 1.0, scratch);
        }

        for (int i = 0; i < _n; i++)
            for (int j = 0; j < _n; j++)
            {
                var d = Math.Abs(atZero[i * _n + j] - atOne[i * _n + j]);
                if (d > 1e-9 * (1 + Math.Abs(atZero[i * _n + j])))
                    throw new InvalidOperationException(
                        $"ImplicitBlock: residual {i} is not affine in eliminated variable x[{Variables[j].Index}] " +
                        $"(∂E_{i}/∂v_{j} = {atZero[i * _n + j]:G6} at v=0 but {atOne[i * _n + j]:G6} at v=1). " +
                        "Constraint expressions passed to AddImplicitBlock must be linear in the eliminated variables.");
            }

        ArrayPool<double>.Shared.Return(atZero);
        ArrayPool<double>.Shared.Return(atOne);
        ArrayPool<double>.Shared.Return(scratch);
        ArrayPool<double>.Shared.Return(gradBuf);
    }

    /// <summary>Zero only the totalVars-buffer entries that <see cref="_residualVarsRaw"/> ever
    /// writes into during a raw-mode AccumulateGradient call. Avoids clearing the full ~17K-double
    /// buffer when only ~25 entries actually get touched. Caller must keep the buffer's other
    /// entries clean (true for all our usages: ArrayPool-rented and AccumulateGradient writes only
    /// at residual var indices).</summary>
    private void ClearAtResidualVars(double[] buffer)
    {
        var ix = _residualVarIndices!;
        for (int i = 0; i < ix.Length; i++) buffer[ix[i]] = 0.0;
    }

    private void ExtractA(double[] outA, ReadOnlySpan<double> scratchSpan, double[] gradBuf, int totalVars, double ownVarValue, double[] scratch)
    {
        for (int j = 0; j < _n; j++) scratch[Variables[j].Index] = ownVarValue;
        for (int i = 0; i < _n; i++)
        {
            Array.Clear(gradBuf, 0, totalVars);
            Residuals[i].AccumulateGradient(scratchSpan, gradBuf);
            for (int j = 0; j < _n; j++)
                outA[i * _n + j] = gradBuf[Variables[j].Index];
        }
    }

    /// <summary>Solves A·v* = b for the current scratch state and writes v* into scratch[Variables[j].Index].
    /// Idempotent within a single eval generation.</summary>
    public void Solve(double[] scratch, long evalGeneration, double[] tempGradBuffer)
    {
        if (_generation == evalGeneration) return;
        _generation = evalGeneration;

        // Phase 1: zero own eliminated vars in scratch (so residuals at v=0 give -b_i).
        for (int j = 0; j < _n; j++)
            scratch[Variables[j].Index] = 0.0;
        Model.InvalidateValueCache();

        // Phase 2: extract b and A in raw mode.
        using (Model.EnterRawMode())
        {
            var scratchSpan = new ReadOnlySpan<double>(scratch);
            for (int i = 0; i < _n; i++)
            {
                _b[i] = -Residuals[i].Evaluate(scratchSpan);

                ClearAtResidualVars(tempGradBuffer);
                Residuals[i].AccumulateGradient(scratchSpan, tempGradBuffer);
                for (int j = 0; j < _n; j++)
                    _A[i * _n + j] = tempGradBuffer[Variables[j].Index];
            }
        }

        // Phase 3: LU factorize A and back-solve for v*.
        Array.Copy(_A, _luFactor, _n * _n);
        Array.Copy(_b, _vstar, _n);
        LuDecompose(_luFactor, _perm, _n);
        LuSolve(_luFactor, _perm, _vstar, _n);

        // Phase 4: write v* into scratch. No scale conversion is needed: A and b were extracted in
        // raw mode, where VariableNode contributes its own Scale, so the linear system is posed in
        // scratch units and v* comes out in them too. VariableNode.Evaluate (= scratch[Index]·Scale)
        // then recovers physical units, exactly as for a non-eliminated variable.
        for (int j = 0; j < _n; j++)
            scratch[Variables[j].Index] = _vstar[j];
        Model.InvalidateValueCache();
    }

    /// <summary>Solves A·x = rhs in place with this block's current LU factors, for a right-hand side
    /// the caller owns. The factors are the ones <see cref="Solve"/> computed for the current
    /// evaluation pass, so this is only meaningful once the block has solved.
    /// <see cref="ReducedDerivatives"/> uses it for the forward substitution that produces ∂v/∂p.</summary>
    internal void SolveWithFactor(Span<double> rhs) => LuSolve(_luFactor, _perm, rhs, _n);

    /// <summary>Transpose counterpart of <see cref="SolveWithFactor"/>: solves Aᵀ·x = rhs in place,
    /// for the adjoint's backward substitution.</summary>
    internal void SolveTransposeWithFactor(Span<double> rhs) => LuSolveTranspose(_luFactor, _perm, rhs, _n);

    /// <summary>Returns sorted array of original Variable.Index values for non-eliminated inputs
    /// of this block. Cached after first call.</summary>
    private int[] InputIndicesSorted
    {
        get
        {
            if (_inputIndicesSorted is null)
            {
                if (_inputs is null)
                {
                    var tmp = new HashSet<Variable>();
                    CollectInputVariables(tmp);
                }
                _inputIndicesSorted = _inputs!.Select(v => v.Index).OrderBy(i => i).ToArray();
            }
            return _inputIndicesSorted;
        }
    }

    /// <summary>Adds the input-clique sparsity contribution this block makes to a Hessian sparsity
    /// pattern: every pair of decision-input indices is a candidate non-zero. Used by
    /// <see cref="VariableNode.CollectHessianSparsity"/> for eliminated VariableNodes — exposed
    /// here as a method so the caller doesn't have to allocate a fresh HashSet&lt;Variable&gt;
    /// per call.</summary>
    public void AddInputCliqueToHessianSparsity(HashSet<(int row, int col)> entries)
    {
        var idx = InputIndicesSorted;
        for (int i = 0; i < idx.Length; i++)
            for (int j = 0; j <= i; j++)
                ExprNode.AddSparsityEntry(entries, idx[i], idx[j]);
    }

    // ----------------- Small dense LU with partial pivoting -----------------

    private static void LuDecompose(double[] A, int[] perm, int n)
    {
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int k = 0; k < n; k++)
        {
            int piv = k;
            double maxAbs = Math.Abs(A[k * n + k]);
            for (int i = k + 1; i < n; i++)
            {
                var v = Math.Abs(A[i * n + k]);
                if (v > maxAbs) { maxAbs = v; piv = i; }
            }
            if (maxAbs < 1e-14)
                throw new InvalidOperationException(
                    $"ImplicitBlock LU: singular system (pivot at row {k} = {maxAbs:E2}). " +
                    "The eliminated subsystem is rank-deficient at the current iterate.");
            if (piv != k)
            {
                (perm[k], perm[piv]) = (perm[piv], perm[k]);
                for (int j = 0; j < n; j++)
                    (A[k * n + j], A[piv * n + j]) = (A[piv * n + j], A[k * n + j]);
            }
            var diag = A[k * n + k];
            // Explicit inner loop instead of TensorPrimitives.MultiplyAdd: at our typical n=3 the
            // tail length (n-k-1) is 0..2, where the call's dispatch overhead dwarfs its SIMD work.
            // The JIT vectorises this loop just as well when the tail is large enough to matter.
            for (int i = k + 1; i < n; i++)
            {
                A[i * n + k] /= diag;
                var factor = A[i * n + k];
                for (int j = k + 1; j < n; j++)
                    A[i * n + j] -= factor * A[k * n + j];
            }
        }
    }

    /// <summary>Maximum block size we stackalloc the LU work buffer for. Above this we fall back
    /// to ArrayPool to avoid blowing the (typically 1 MB) stack frame. 256 doubles = 2 KB on the
    /// stack — generous for the typical few-eliminated-vars-per-block usage and still safe.</summary>
    private const int LuWorkStackallocThreshold = 256;

    private static void LuSolve(double[] LU, int[] perm, Span<double> bx, int n)
    {
        // For the typical n (block size, e.g. 3 in the inferrer) we stackalloc — the rent/return
        // overhead vs. the actual work isn't worth it. Above the threshold, fall back to pool.
        // Explicit scalar inner loops (no TensorPrimitives.Dot): at small n the dispatch overhead
        // of TensorPrimitives swamps its SIMD win, and the JIT vectorises these loops just fine.
        double[]? rented = null;
        Span<double> work = n <= LuWorkStackallocThreshold
            ? stackalloc double[LuWorkStackallocThreshold]
            : (rented = ArrayPool<double>.Shared.Rent(n));
        work = work[..n];
        for (int i = 0; i < n; i++) work[i] = bx[perm[i]];
        for (int i = 0; i < n; i++)
        {
            double sum = work[i];
            for (int j = 0; j < i; j++)
                sum -= LU[i * n + j] * work[j];
            work[i] = sum;
        }
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = work[i];
            for (int j = i + 1; j < n; j++)
                sum -= LU[i * n + j] * work[j];
            work[i] = sum / LU[i * n + i];
        }
        for (int i = 0; i < n; i++) bx[i] = work[i];
        if (rented is not null) ArrayPool<double>.Shared.Return(rented);
    }

    private static void LuSolveTranspose(double[] LU, int[] perm, Span<double> bx, int n)
    {
        // Solve A^T x = bx where A = P^-1 L U (P from `perm`), so A^T = U^T L^T P
        // Step 1: U^T y = bx
        // Step 2: L^T z = y
        // Step 3: x = P^-1 z, i.e. x[perm[i]] = z[i]
        // Same stackalloc-or-pool pattern as LuSolve (see LuWorkStackallocThreshold).
        double[]? rented = null;
        Span<double> work = n <= LuWorkStackallocThreshold
            ? stackalloc double[LuWorkStackallocThreshold]
            : (rented = ArrayPool<double>.Shared.Rent(n));
        work = work[..n];
        for (int i = 0; i < n; i++)
        {
            var sum = bx[i];
            for (int j = 0; j < i; j++)
                sum -= LU[j * n + i] * work[j];
            work[i] = sum / LU[i * n + i];
        }
        for (int i = n - 1; i >= 0; i--)
        {
            var sum = work[i];
            for (int j = i + 1; j < n; j++)
                sum -= LU[j * n + i] * work[j];
            work[i] = sum;
        }
        for (int i = 0; i < n; i++) bx[perm[i]] = work[i];
        if (rented is not null) ArrayPool<double>.Shared.Return(rented);
    }
}
