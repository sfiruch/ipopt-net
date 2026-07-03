namespace IpoptNet.Modelling;

/// <summary>
/// Abstract accumulator for sparse second-order derivatives. Concrete variants:
///   - <see cref="SparseHessianAccumulator"/>: CSR-backed, sparsity declared up front. Used for
///     the model-level Hessian IPOPT consumes.
///   - <see cref="DenseLocalHessianAccumulator"/>: dense local matrix indexed by Variable.Index,
///     entries created on demand. Used internally (e.g. by ImplicitBlock) to compute residual
///     Hessians in raw mode without preallocating a full CSR.
/// </summary>
public abstract class HessianAccumulator
{
    public abstract void Add(int i, int j, double value);
    public abstract double Get(int i, int j);
    public abstract void Clear();
    public abstract ReadOnlySpan<double> Values { get; }

    /// <summary>Resolves (i, j) to a stable slot in this accumulator's value storage, so hot
    /// loops with a fixed sparsity footprint can resolve their entries once and use
    /// <see cref="AddAtSlot"/> afterwards, skipping the per-Add index lookup. Slots stay valid
    /// for the accumulator's lifetime; callers cache them keyed on the accumulator instance.</summary>
    public abstract int GetSlot(int i, int j);
    public abstract void AddAtSlot(int slot, double value);
}

/// <summary>
/// CSR-backed Hessian accumulator with up-front sparsity declaration. <see cref="Add"/> requires
/// (i, j) to be in the predeclared structure passed to the constructor.
/// </summary>
public sealed class SparseHessianAccumulator : HessianAccumulator
{
    private readonly double[] _values;
    private readonly int[] _rowPointers;
    private readonly int[] _colIndices;

    public SparseHessianAccumulator(int n, int[] structRows, int[] structCols)
    {
        var nnz = structRows.Length;
        _values = new double[nnz];
        _colIndices = new int[nnz];
        _rowPointers = new int[n + 1];

        for (int i = 0; i < nnz; i++)
        {
            _colIndices[i] = structCols[i];
            _rowPointers[structRows[i] + 1]++;
        }
        for (int i = 1; i <= n; i++)
            _rowPointers[i] += _rowPointers[i - 1];
    }

    public override void Add(int i, int j, double value)
    {
        if (i < j) (i, j) = (j, i);
        int start = _rowPointers[i];
        int length = _rowPointers[i + 1] - start;
        int idx = Array.BinarySearch(_colIndices, start, length, j);
        _values[idx] += value;
    }

    public override double Get(int i, int j)
    {
        if (i < j) (i, j) = (j, i);
        int start = _rowPointers[i];
        int length = _rowPointers[i + 1] - start;
        int idx = Array.BinarySearch(_colIndices, start, length, j);
        return idx >= 0 ? _values[idx] : 0.0;
    }

    public override int GetSlot(int i, int j)
    {
        if (i < j) (i, j) = (j, i);
        int start = _rowPointers[i];
        return Array.BinarySearch(_colIndices, start, _rowPointers[i + 1] - start, j);
    }

    public override void AddAtSlot(int slot, double value) => _values[slot] += value;

    public override ReadOnlySpan<double> Values => _values;

    public override void Clear() => Array.Clear(_values);
}

/// <summary>
/// Dense Hessian accumulator backed by an n × n matrix indexed by *local* positions. A Variable.Index
/// → local-index map is built from the indices passed to the constructor. Used by ImplicitBlock
/// to compute the per-residual Hessian (∂²E_l/∂y_a∂y_b for all pairs in the residual's variable
/// set) without allocating a totalVars-sized CSR.
/// </summary>
public sealed class DenseLocalHessianAccumulator : HessianAccumulator
{
    private readonly Dictionary<int, int> _origToLocal;
    private readonly int _n;
    private readonly double[] _matrix;  // n × n, row-major; only lower triangle is meaningful

    public DenseLocalHessianAccumulator(IReadOnlyList<int> originalIndices)
    {
        _n = originalIndices.Count;
        _origToLocal = new Dictionary<int, int>(_n);
        for (int i = 0; i < _n; i++)
            _origToLocal[originalIndices[i]] = i;
        _matrix = new double[_n * _n];
    }

    public override void Add(int i, int j, double value)
    {
        var li = _origToLocal[i];
        var lj = _origToLocal[j];
        if (li < lj) (li, lj) = (lj, li);
        _matrix[li * _n + lj] += value;
    }

    public override double Get(int i, int j)
    {
        if (!_origToLocal.TryGetValue(i, out var li)) return 0.0;
        if (!_origToLocal.TryGetValue(j, out var lj)) return 0.0;
        if (li < lj) (li, lj) = (lj, li);
        return _matrix[li * _n + lj];
    }

    /// <summary>Direct local-index access (faster than <see cref="Get"/> when callers already know
    /// the local positions).</summary>
    public double GetByLocal(int li, int lj)
    {
        if (li < lj) (li, lj) = (lj, li);
        return _matrix[li * _n + lj];
    }

    public override int GetSlot(int i, int j)
    {
        var li = _origToLocal[i];
        var lj = _origToLocal[j];
        if (li < lj) (li, lj) = (lj, li);
        return li * _n + lj;
    }

    public override void AddAtSlot(int slot, double value) => _matrix[slot] += value;

    public override void Clear() => Array.Clear(_matrix);

    public override ReadOnlySpan<double> Values => _matrix;

    public IReadOnlyDictionary<int, int> OriginalToLocal => _origToLocal;
    public int Size => _n;
}
