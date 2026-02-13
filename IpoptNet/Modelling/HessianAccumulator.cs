using System.Runtime.InteropServices;

namespace IpoptNet.Modelling;

public sealed class HessianAccumulator
{
    public readonly Dictionary<(int Row, int Col), double> Entries = new();

    public HessianAccumulator() { }

    public void Add(int i, int j, double value)
    {
        if (Math.Abs(value) < 1e-18)
            return;

        var key = i >= j ? (i, j) : (j, i);
        ref var entry = ref CollectionsMarshal.GetValueRefOrAddDefault(Entries, key, out _);
        entry += value;
    }

    public void Clear() => Entries.Clear();
}
