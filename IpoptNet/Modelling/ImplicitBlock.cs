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

    /// <summary>Per-pass cache of sensitivity vectors. _sensCache[j][k] = ∂v*_j/∂x_decision
    /// where x_decision = the variable at original index _inputIndicesSorted[k]. Decision vars
    /// only — eliminated inputs are chained into via recursive cached lookup. Without this cache,
    /// recursion across chained blocks blows up exponentially in chain depth.</summary>
    private double[]?[]? _sensCache;
    private long _sensGeneration = -1;

    /// <summary>Sorted array of original Variable.Index values for decision-vector inputs of this
    /// block (transitive across other blocks). Built from _inputs on first need.</summary>
    private int[]? _inputIndicesSorted;

    // ----- Per-pass Hessian-sensitivity caching -----
    // Computed once per evaluation generation when a Hessian query first hits this block.
    // _yVars: residual_vars_raw minus this block's own eliminated vars.
    // _residualHessians[l]: dense K × K (where K = _residualVarsRaw.Length) raw-mode Hessian of residual l, at v=v*.
    // _sLocal[j][a]: ∂v*_j/∂y_a (a indexes into _yVars).
    // _tLocal[j][a*M + b]: ∂²v*_j/∂y_a∂y_b (M = _yVars.Length, symmetric).
    // _hess2Cache[j][k*N + p]: ∂²v*_j/∂x_dec_k∂x_dec_p (N = inputs.Length, symmetric, decision-var space).
    //
    // Storage is allocated once on first need (when shapes K, M become known) and reused across
    // every subsequent evaluation pass — only the *values* are refreshed per pass. Avoids the
    // per-pass GC pressure of allocating ~9 double[] of varying shapes per block × 5760 blocks.
    private Variable[]? _yVars;
    private int[]? _yPosInRaw;
    private int[]? _ownPosInRaw;
    private int _M, _K;                      // sizes after first init; const for the block's lifetime
    private int[]? _rawIndicesArr;
    private DenseLocalHessianAccumulator? _localAcc;
    private double[]? _rhs;                  // [_n]
    private double[][]? _residualHessians;   // [l][a*K + b], symmetric
    private double[][]? _rawGradArr;         // [l][a] of size M
    private double[][]? _sLocal;             // [j][a]
    private double[][]? _tLocal;             // [j][a*M + b], symmetric
    private long _localHessGen = -1;
    private double[][]? _hess2Cache;         // [j][k*N + p], symmetric — preallocated, gated by _hess2Computed
    private bool[]? _hess2Computed;          // [j]; reset on generation change
    private double[]? _muFlat;               // [a*N + k] — reused across passes
    private double[]? _qFlat;                // [a*N + k] per j; reused across passes
    private int _N;                          // = InputIndicesSorted.Length, captured at first GetSecondOrderSensitivity
    private int[][]? _otherKMapPerY;         // [a][kPos] — fixed once known; -1 if inputs[kPos] not in y_a's block's inputs
    private int[]? _otherNPerY;              // [a] = other block's input count
    private int[]? _directKPosPerY;          // [a] = position of direct-decision y_a in inputs (-1 for other-elim)
    private bool[]? _otherMapIsIdentityPerY; // [a] = true if otherN==N and map is identity (enables contiguous MultiplyAdd in ν chain)
    private long _hess2Gen = -1;
    private bool _computingHess2;            // recursion guard for second-order sensitivity
    private bool _computingSens;             // recursion guard for first-order sensitivity

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
        using (Model.EnterRawMode())
            foreach (var r in Residuals)
                r.Prepare();

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

    /// <summary>Computes the transitive closure of decision-vector variables this block depends on,
    /// for upstream sparsity analysis. Excludes this block's own eliminated variables.</summary>
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
                    else
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

        // Phase 4: write v* into scratch. Eliminated vars must have Scale==1 (enforced at
        // AddImplicitBlock); VariableNode.Evaluate (= scratch[Index] * Scale) returns physical units.
        // Lifting the Scale==1 mandate would require dividing v* by Scale here so the existing
        // VariableNode.Evaluate path stays correct.
        for (int j = 0; j < _n; j++)
            scratch[Variables[j].Index] = _vstar[j];
    }

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

    /// <summary>Returns ∂v*_j/∂x_decision_k where k indexes into InputIndicesSorted. Computed once
    /// per evaluation pass and cached. Caching breaks the exponential recursion across chained
    /// blocks: each (block, j) pair is computed at most once per pass.</summary>
    private double[] GetSensitivity(int indexInBlock, ReadOnlySpan<double> scratch)
    {
        if (_sensGeneration != _generation)
        {
            _sensCache = new double[_n][];
            _sensGeneration = _generation;
        }
        if (_sensCache![indexInBlock] is { } cached) return cached;

        if (_computingSens)
            throw new InvalidOperationException(
                "ImplicitBlock: cycle detected in GetSensitivity. Implicit blocks must be added in topological order " +
                "(checked at AddImplicitBlock; this exception means the topological-order check missed a case).");
        _computingSens = true;
        try
        {
            var inputs = InputIndicesSorted;
            var S = new double[inputs.Length];
            var lambda = ArrayPool<double>.Shared.Rent(_n);
            var totalVars = scratch.Length;
            var gradBuf = ArrayPool<double>.Shared.Rent(totalVars);

            Array.Clear(lambda, 0, _n);
            lambda[indexInBlock] = -1.0;
            LuSolveTranspose(_luFactor, _perm, lambda, _n);

            // gradBuf_k = Σ_l λ_l · ∂E_l/∂x_k|v=v*  (raw mode). Selective clear: AccumulateGradient
            // in raw mode only writes to indices in _residualVarsRaw, so zero only those entries
            // — clearing the whole totalVars-sized buffer was 30%+ of total profile time.
            ClearAtResidualVars(gradBuf);
            using (Model.EnterRawMode())
                for (int l = 0; l < _n; l++)
                    Residuals[l].AccumulateGradient(scratch, gradBuf.AsSpan(0, totalVars), lambda[l]);

            // Distribute:
            //   - own elim vars: skip
            //   - other-block elim vars: chain via cached sensitivity (other's inputs ⊆ ours)
            //   - decision vars: write directly via BinarySearch into our InputIndicesSorted
            foreach (var v in _residualVarsRaw!)
            {
                var seed = gradBuf[v.Index];
                if (seed == 0.0) continue;
                if (v.Block == this) continue;
                if (v.Block is { } other)
                {
                    var otherS = other.GetSensitivity(v.IndexInBlock, scratch);
                    var otherInputs = other.InputIndicesSorted;
                    for (int k = 0; k < otherS.Length; k++)
                    {
                        if (otherS[k] == 0.0) continue;
                        int pos = Array.BinarySearch(inputs, otherInputs[k]);
                        if (pos < 0)
                            throw new InvalidOperationException(
                                $"ImplicitBlock: input variable x[{otherInputs[k]}] from chained block missing from this block's input set. " +
                                "CollectInputVariables transitive closure is incomplete.");
                        S[pos] += seed * otherS[k];
                    }
                }
                else
                {
                    int pos = Array.BinarySearch(inputs, v.Index);
                    if (pos < 0)
                        throw new InvalidOperationException(
                            $"ImplicitBlock: decision variable x[{v.Index}] referenced by residual but not in InputIndicesSorted. " +
                            "CollectInputVariables is missing this variable.");
                    S[pos] += seed;
                }
            }

            ArrayPool<double>.Shared.Return(lambda);
            ArrayPool<double>.Shared.Return(gradBuf);

            _sensCache[indexInBlock] = S;
            return S;
        }
        finally { _computingSens = false; }
    }

    /// <summary>Reverse-mode gradient propagation through this block. Seed `weight` is ∂upstream/∂v_j
    /// where j = indexInBlock. Adds w · ∂v_j/∂x_k to compactGrad for each non-eliminated input variable
    /// x_k via the precomputed cached sensitivity.</summary>
    public void PropagateGradient(int indexInBlock, ReadOnlySpan<double> scratch, Span<double> compactGrad, double weight, int[] callerSortedVarIndices)
    {
        var S = GetSensitivity(indexInBlock, scratch);
        var inputs = InputIndicesSorted;
        for (int k = 0; k < S.Length; k++)
        {
            var s = S[k];
            if (s == 0.0) continue;
            int pos = Array.BinarySearch(callerSortedVarIndices, inputs[k]);
            if (pos < 0)
                throw new InvalidOperationException(
                    $"ImplicitBlock.PropagateGradient: input x[{inputs[k]}] not in caller's sorted-var-indices. " +
                    "CollectVariables on the upstream expression must include this block's transitive inputs.");
            compactGrad[pos] += weight * s;
        }
    }

    /// <summary>Computes and caches the per-residual raw-mode Hessian M_l[a, b] = ∂²E_l/∂y_a∂y_b
    /// (at v=v*), the per-residual local first-order sensitivity S_local[j, a] = ∂v*_j/∂y_a, and
    /// the local second-order sensitivity T_local[j, a, b] = ∂²v*_j/∂y_a∂y_b — all in y-space
    /// (residual_vars_raw with own elim vars excluded for the y indexing).</summary>
    private void EnsureLocalHessians(ReadOnlySpan<double> scratch)
    {
        if (_localHessGen == _generation) return;
        _localHessGen = _generation;

        // One-time-per-block storage init: shapes (K, M) and all the dense buffers below are
        // constant after PrepareResiduals, so allocate on first call and reuse forever.
        if (_residualHessians is null)
        {
            int K = _residualVarsRaw!.Length;
            _K = K;
            var ownSet = new HashSet<Variable>(Variables);
            var yList = new List<Variable>(K);
            var rawPos = new Dictionary<int, int>(K);
            for (int i = 0; i < K; i++)
            {
                rawPos[_residualVarsRaw[i].Index] = i;
                if (!ownSet.Contains(_residualVarsRaw[i]))
                    yList.Add(_residualVarsRaw[i]);
            }
            _yVars = yList.ToArray();
            int M = _yVars.Length;
            _M = M;

            _yPosInRaw = new int[M];
            for (int i = 0; i < M; i++) _yPosInRaw[i] = rawPos[_yVars[i].Index];

            _ownPosInRaw = new int[_n];
            for (int j = 0; j < _n; j++) _ownPosInRaw[j] = rawPos[Variables[j].Index];

            _rawIndicesArr = new int[K];
            for (int i = 0; i < K; i++) _rawIndicesArr[i] = _residualVarsRaw[i].Index;
            _localAcc = new DenseLocalHessianAccumulator(_rawIndicesArr);

            _residualHessians = new double[_n][];
            _rawGradArr = new double[_n][];
            _sLocal = new double[_n][];
            _tLocal = new double[_n][];
            for (int j = 0; j < _n; j++)
            {
                _residualHessians[j] = new double[K * K];
                _rawGradArr[j] = new double[M];
                _sLocal[j] = new double[M];
                _tLocal[j] = new double[M * M];
            }
            _rhs = new double[_n];
        }

        var rawIndices = _rawIndicesArr!;
        var localAcc = _localAcc!;
        var rhs = _rhs!;
        var Mlen = _M;
        var Klen = _K;

        // 1) Per-residual raw Hessian — refresh values into the persistent storage. Read directly
        // from localAcc's internal lower-triangular matrix (exposed via Values) instead of going
        // through Get(originalIdx, originalIdx) which does two Dictionary lookups + a swap per
        // entry. We construct localAcc with rawIndices in 0..K-1 order, so the local index of
        // _residualVarsRaw[i] is exactly i.
        using (Model.EnterRawMode())
        {
            for (int l = 0; l < _n; l++)
            {
                localAcc.Clear();
                Residuals[l].AccumulateHessian(scratch, localAcc, 1.0);
                var H = _residualHessians[l];
                var src = localAcc.Values;  // K × K row-major, only lower triangle (j ≤ i) is valid
                // Copy lower triangle row-by-row, then mirror to upper.
                for (int i = 0; i < Klen; i++)
                    src.Slice(i * Klen, i + 1).CopyTo(H.AsSpan(i * Klen, i + 1));
                for (int i = 0; i < Klen; i++)
                    for (int j = 0; j < i; j++)
                        H[j * Klen + i] = H[i * Klen + j];
            }
        }

        // 2) Per-residual raw gradients into _rawGradArr[l] (length M).
        var totalVars = scratch.Length;
        var gradBuf = ArrayPool<double>.Shared.Rent(totalVars);
        using (Model.EnterRawMode())
        {
            for (int l = 0; l < _n; l++)
            {
                ClearAtResidualVars(gradBuf);
                Residuals[l].AccumulateGradient(scratch, gradBuf.AsSpan(0, totalVars));
                var arr = _rawGradArr![l];
                for (int a = 0; a < Mlen; a++)
                    arr[a] = gradBuf[_yVars![a].Index];
            }
        }
        ArrayPool<double>.Shared.Return(gradBuf);

        // 3) Solve for S_local[:, a] = -A^-1 · rawGrad[:, a] for each y_a.
        for (int a = 0; a < Mlen; a++)
        {
            for (int l = 0; l < _n; l++) rhs[l] = _rawGradArr![l][a];
            LuSolve(_luFactor, _perm, rhs, _n);
            for (int j = 0; j < _n; j++) _sLocal![j][a] = -rhs[j];
        }

        // 4) Solve for T_local[:, a, b] = -A^-1 · RHS[:, a, b] for each (a, b) pair (a ≤ b).
        //    RHS_l[a, b] = M_l[y_a, y_b] + Σ_c M_l[v_c, y_b] · S_local[c, a] + Σ_c M_l[v_c, y_a] · S_local[c, b]
        for (int a = 0; a < Mlen; a++)
        {
            int a_raw = _yPosInRaw![a];
            for (int b = a; b < Mlen; b++)
            {
                int b_raw = _yPosInRaw[b];
                for (int l = 0; l < _n; l++)
                {
                    var H_l = _residualHessians[l];
                    double v = H_l[a_raw * Klen + b_raw];
                    for (int c = 0; c < _n; c++)
                    {
                        int c_raw = _ownPosInRaw![c];
                        v += H_l[c_raw * Klen + b_raw] * _sLocal![c][a];
                        v += H_l[c_raw * Klen + a_raw] * _sLocal![c][b];
                    }
                    rhs[l] = v;
                }
                LuSolve(_luFactor, _perm, rhs, _n);
                for (int j = 0; j < _n; j++)
                {
                    var t = -rhs[j];
                    _tLocal![j][a * Mlen + b] = t;
                    if (a != b) _tLocal[j][b * Mlen + a] = t;
                }
            }
        }
    }

    /// <summary>Returns ∂²v*_j/∂x_dec_k∂x_dec_p (decision-variable-space Hessian sensitivity), indexed
    /// as flat [k_pos * N + p_pos] with N = InputIndicesSorted.Length. Computed lazily, cached per
    /// pass. Uses chain rule: T_decision[k, p] = Σ_{a,b} T_local[a, b] · μ_a[k] · μ_b[p]
    ///                                          + Σ_a S_local[a] · ν_a[k, p]
    /// where μ_a[k] = ∂y_a/∂x_dec_k (1 for direct decision, S_other[a, k] for chained other-elim),
    /// and ν_a[k, p] = ∂²y_a/∂x_dec_k∂x_dec_p (0 for decision, T_other[a, k, p] for chained).</summary>
    public double[] GetSecondOrderSensitivity(int indexInBlock, ReadOnlySpan<double> scratch)
    {
        if (_hess2Gen != _generation)
        {
            // New evaluation pass — invalidate per-j cache without reallocating.
            _hess2Computed?.AsSpan().Clear();
            _hess2Gen = _generation;
        }
        if (_hess2Cache is not null && _hess2Computed![indexInBlock])
            return _hess2Cache[indexInBlock];

        if (_computingHess2)
            throw new InvalidOperationException("ImplicitBlock: cycle detected in GetSecondOrderSensitivity. Blocks must be added in topological order.");
        _computingHess2 = true;
        try
        {
            EnsureLocalHessians(scratch);
            // Make sure all S[*] for this block are computed (used below for ν chain via other blocks).
            for (int j = 0; j < _n; j++) GetSensitivity(j, scratch);

            var inputs = InputIndicesSorted;
            int N = inputs.Length;
            int M = _M;

            // One-time alloc of per-block buffers and the (a → otherK[]) lookup maps.
            if (_hess2Cache is null)
            {
                _N = N;
                _hess2Cache = new double[_n][];
                _hess2Computed = new bool[_n];
                for (int j = 0; j < _n; j++) _hess2Cache[j] = new double[N * N];
                _muFlat = new double[M * N];
                _qFlat = new double[M * N];

                // Precompute, for each y_a, the otherK[kPos] mapping into the chained block's
                // input space — these are *fixed* once both blocks are prepared, so cache once.
                _otherKMapPerY = new int[M][];
                _otherNPerY = new int[M];
                _directKPosPerY = new int[M];
                _otherMapIsIdentityPerY = new bool[M];
                for (int a = 0; a < M; a++)
                {
                    var Ya = _yVars![a];
                    if (Ya.Block is null)
                    {
                        int pos = Array.BinarySearch(inputs, Ya.Index);
                        if (pos < 0)
                            throw new InvalidOperationException($"ImplicitBlock: y-decision var x[{Ya.Index}] missing from inputs.");
                        _directKPosPerY[a] = pos;
                        _otherKMapPerY[a] = null!;
                        _otherNPerY[a] = 0;
                    }
                    else
                    {
                        _directKPosPerY[a] = -1;
                        var otherInputs = Ya.Block.InputIndicesSorted;
                        var map = new int[N];
                        bool isIdentity = otherInputs.Length == N;
                        for (int kPos = 0; kPos < N; kPos++)
                        {
                            map[kPos] = Array.BinarySearch(otherInputs, inputs[kPos]);
                            if (map[kPos] != kPos) isIdentity = false;
                        }
                        _otherKMapPerY[a] = map;
                        _otherNPerY[a] = otherInputs.Length;
                        _otherMapIsIdentityPerY[a] = isIdentity;
                    }
                }
            }

            // Refresh μ for this pass. Per a:
            //   - direct-decision y_a: μ[a][directKPos] = 1, others = 0.
            //   - other-elim y_a:      μ[a][kPos] = otherS[otherKMap[a][kPos]] (or 0 if unmapped).
            var muFlat = _muFlat!;
            Array.Clear(muFlat, 0, M * N);
            for (int a = 0; a < M; a++)
            {
                var Ya = _yVars![a];
                if (Ya.Block is null)
                {
                    muFlat[a * N + _directKPosPerY![a]] = 1.0;
                }
                else
                {
                    var otherS = Ya.Block.GetSensitivity(Ya.IndexInBlock, scratch);
                    var map = _otherKMapPerY![a];
                    for (int kPos = 0; kPos < N; kPos++)
                    {
                        int otherK = map[kPos];
                        if (otherK >= 0) muFlat[a * N + kPos] = otherS[otherK];
                    }
                }
            }

            // Outer-product matmul reorder: instead of computing each Q[a, k] / T[k, p] entry
            // via an inner sum that strides through a column (cache-unfriendly, hard to vectorise),
            // accumulate row contributions:
            //   Q[a, ·] += T_local[a, b] · μ[b, ·]              (inner loop over k, contiguous N-vector)
            //   T[k, ·] += μ[a, k] · Q[a, ·] + sLa · ν[a, k, ·] (inner loop over p, contiguous N-vector)
            // Both inner loops touch only contiguous spans of qFlat/μ/Q/Tj — the JIT auto-vectorises
            // them. Wins over the natural sum-then-store ordering for our small M (~25), where
            // TensorPrimitives.Dot's dispatch overhead doesn't amortise.
            var qFlat = _qFlat!;
            for (int j = 0; j < _n; j++)
            {
                var Tj = _hess2Cache[j];
                var TLoc_j = _tLocal![j];
                var sLoc_j = _sLocal![j];

                // Q = T_local · μ, accumulated outer-product style.
                Array.Clear(qFlat, 0, M * N);
                for (int a = 0; a < M; a++)
                {
                    var qRowA = qFlat.AsSpan(a * N, N);
                    for (int b = 0; b < M; b++)
                    {
                        double tab = TLoc_j[a * M + b];
                        if (tab == 0.0) continue;
                        TensorPrimitives.MultiplyAdd<double>(muFlat.AsSpan(b * N, N), tab, qRowA, qRowA);
                    }
                }

                // T[k, ·] = Σ_a (μ[a, k] · Q[a, ·]) + Σ_a (S_local[a] · ν[a, k, ·])
                // ν is NOT materialised — looked up per (a, k) via cached otherKMap → otherT.
                Array.Clear(Tj, 0, N * N);
                for (int a = 0; a < M; a++)
                {
                    var qRowA = qFlat.AsSpan(a * N, N);
                    var Ya = _yVars![a];
                    var otherBlock = Ya.Block;
                    var sLa = sLoc_j[a];
                    bool hasNu = otherBlock is not null && sLa != 0.0;
                    int[]? map = hasNu ? _otherKMapPerY![a] : null;
                    double[]? otherT = hasNu ? otherBlock!.GetSecondOrderSensitivity(Ya.IndexInBlock, scratch) : null;
                    int otherN = hasNu ? _otherNPerY![a] : 0;
                    bool mapIsIdentity = hasNu && _otherMapIsIdentityPerY![a];

                    for (int kPos = 0; kPos < N; kPos++)
                    {
                        var tRowK = Tj.AsSpan(kPos * N, N);
                        double mak = muFlat[a * N + kPos];
                        if (mak != 0.0)
                            TensorPrimitives.MultiplyAdd<double>(qRowA, mak, tRowK, tRowK);

                        if (hasNu)
                        {
                            int otherK = map![kPos];
                            if (otherK < 0) continue;
                            var otherTRowK = otherT.AsSpan(otherK * otherN, otherN);
                            if (mapIsIdentity)
                            {
                                // Map is the identity (otherInputs == inputs as sorted lists, otherN == N).
                                // Then otherP == pPos, so the inner becomes a contiguous AXPY:
                                //   tRowK[pPos] += sLa · otherTRowK[pPos]
                                // For the inferrer this is the common case (every block depends on
                                // the same parameters + initial conditions) and saves the gather.
                                TensorPrimitives.MultiplyAdd<double>(otherTRowK[..N], sLa, tRowK, tRowK);
                            }
                            else
                            {
                                for (int pPos = 0; pPos < N; pPos++)
                                {
                                    int otherP = map[pPos];
                                    if (otherP >= 0) tRowK[pPos] += sLa * otherTRowK[otherP];
                                }
                            }
                        }
                    }
                }

                _hess2Computed![j] = true;
            }

            return _hess2Cache[indexInBlock];
        }
        finally { _computingHess2 = false; }
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

    /// <summary>Reverse-mode Hessian propagation through this block. Adds w · ∂²v_j/∂x_dec_k∂x_dec_p
    /// to hess for each (k, p) pair in the block's decision-input set.</summary>
    public void PropagateHessian(int indexInBlock, ReadOnlySpan<double> scratch, HessianAccumulator hess, double weight)
    {
        var T = GetSecondOrderSensitivity(indexInBlock, scratch);
        var inputs = InputIndicesSorted;
        int N = inputs.Length;
        for (int k = 0; k < N; k++)
        {
            for (int p = 0; p <= k; p++)
            {
                var t = T[k * N + p];
                if (t == 0.0) continue;
                hess.Add(inputs[k], inputs[p], weight * t);
            }
        }
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

    private static void LuSolve(double[] LU, int[] perm, double[] bx, int n)
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

    private static void LuSolveTranspose(double[] LU, int[] perm, double[] bx, int n)
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
