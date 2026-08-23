namespace IpoptNet.Modelling;

/// <summary>
/// Working buffers shared by every <see cref="ImplicitBlock"/> in a model, for the dense
/// intermediates of the second-order sensitivity computation.
///
/// These used to be allocated per block and kept for the block's lifetime, which is the single
/// largest cost in a model with many blocks: a per-step discretisation allocates one block per
/// step, and the buffers scale with the block's own size and its input count. On a 34,558-step fit
/// they came to roughly 250 MB, live for the whole solve.
///
/// Sharing them is safe because they never outlive one call:
/// <list type="bullet">
/// <item><see cref="ImplicitBlock.GetSecondOrderSensitivity"/> computes every j in a single pass and
/// marks them all cached, so it runs its body at most once per block per evaluation generation.</item>
/// <item>Its nested lookups into chained blocks' second-order sensitivities always hit those blocks'
/// own caches, so the recursion never re-enters the body — two blocks are never inside it at once.
/// That is by construction, not luck: the body warms every chained block it will read BEFORE it
/// borrows anything here, and it must stay that way. Reaching a chained block for the first time
/// part-way through a computation lets the nested call overwrite the caller's buffers, and since
/// every buffer is written before it is read (below), the caller then carries on with the callee's
/// numbers and returns silently wrong derivatives rather than failing.</item>
/// <item>Every buffer is fully overwritten before it is read, which the previous per-block version
/// already relied on to reuse them across passes without clearing.</item>
/// </list>
/// Only the memoised results — the per-block first- and second-order sensitivity caches — stay
/// per-block, because those genuinely are read after the call returns.
///
/// Buffers grow to the largest block that has asked for them and are never shrunk. Contents are
/// undefined on return.
/// </summary>
internal sealed class BlockScratch
{
    private double[][]? _residualHessians, _rawGrad, _sLocal, _tLocal;
    private double[]? _rhs, _muFlat, _qFlat, _accumulatorMatrix, _hessSquare, _expandedOther;

    internal double[][] ResidualHessians(int rows, int length) => Rent(ref _residualHessians, rows, length);
    internal double[][] RawGradients(int rows, int length) => Rent(ref _rawGrad, rows, length);
    internal double[][] LocalFirstOrder(int rows, int length) => Rent(ref _sLocal, rows, length);
    internal double[][] LocalSecondOrder(int rows, int length) => Rent(ref _tLocal, rows, length);
    internal double[] Rhs(int length) => Rent(ref _rhs, length);
    internal double[] MuFlat(int length) => Rent(ref _muFlat, length);
    internal double[] QFlat(int length) => Rent(ref _qFlat, length);
    internal double[] AccumulatorMatrix(int length) => Rent(ref _accumulatorMatrix, length);

    /// <summary>Full N × N square a block accumulates its second-order sensitivity into before the
    /// symmetric half of it is copied to the block's own compact store.</summary>
    internal double[] HessianSquare(int length) => Rent(ref _hessSquare, length);

    /// <summary>Full N × N expansion of *another* block's compact second-order sensitivity, so the
    /// nu-chain inner loop keeps reading whole contiguous rows.</summary>
    internal double[] ExpandedOther(int length) => Rent(ref _expandedOther, length);

    private static double[] Rent(ref double[]? store, int length) =>
        store = store is { } s && s.Length >= length ? s : new double[length];

    // Blocks in one model do differ in shape — a step carrying container or grid-flow terms has more
    // residual variables than a bare thermal step — so this genuinely regrows, and every borrower
    // indexes with its own stride into a buffer sized for the largest seen. Rows and row length grow
    // independently: a later block may need more rows of a shorter length, or the reverse, and
    // neither may shrink what an earlier block sized.
    private static double[][] Rent(ref double[][]? store, int rows, int length)
    {
        if (store is { } s && s.Length >= rows && s[0].Length >= length)
            return s;
        rows = Math.Max(rows, store?.Length ?? 0);
        length = Math.Max(length, store is null ? 0 : store[0].Length);
        var grown = new double[rows][];
        for (int i = 0; i < rows; i++)
            grown[i] = new double[length];
        return store = grown;
    }
}
