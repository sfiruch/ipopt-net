namespace IpoptNet.Modelling;

public sealed class HessianAccumulator
{
    private readonly double[] _values;
    private readonly int[] _rowPointers;
    private readonly int[] _colIndices;

    public HessianAccumulator(int n, int[] structRows, int[] structCols)
    {
        var nnz = structRows.Length;
        _values = new double[nnz];
        _colIndices = new int[nnz];
        _rowPointers = new int[n + 1];

        // Build CSR from sorted (row, col) pairs
        for (int i = 0; i < nnz; i++)
        {
            _colIndices[i] = structCols[i];
            _rowPointers[structRows[i] + 1]++;
        }
        // Prefix sum
        for (int i = 1; i <= n; i++)
            _rowPointers[i] += _rowPointers[i - 1];
    }

    public void Add(int i, int j, double value)
    {
        if (i < j)
            (i, j) = (j, i);
        int start = _rowPointers[i];
        int length = _rowPointers[i + 1] - start;
        int idx = Array.BinarySearch(_colIndices, start, length, j);
        _values[idx] += value;
    }

    public double Get(int i, int j)
    {
        if (i < j)
            (i, j) = (j, i);
        int start = _rowPointers[i];
        int length = _rowPointers[i + 1] - start;
        int idx = Array.BinarySearch(_colIndices, start, length, j);
        return idx >= 0 ? _values[idx] : 0.0;
    }

    public ReadOnlySpan<double> Values => _values;

    public void Clear() => Array.Clear(_values);
}
