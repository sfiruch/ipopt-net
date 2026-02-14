namespace IpoptNet.Modelling;

public sealed class HessianAccumulator
{
    private readonly double[] _values;

    public HessianAccumulator(int n)
    {
        _values = new double[n * (n + 1) / 2];
    }

    public void Add(int i, int j, double value)
    {
        // Ensure i >= j for lower triangular storage
        if (i < j)
            (i, j) = (j, i);
        _values[i * (i + 1) / 2 + j] += value;
    }

    public double Get(int i, int j)
    {
        // Ensure i >= j for lower triangular storage
        if (i < j)
            (i, j) = (j, i);
        return _values[i * (i + 1) / 2 + j];
    }

    public void Clear() => Array.Clear(_values);
}
