using System.Runtime.InteropServices;

namespace IpoptNet;

public sealed class IpoptSolver : IDisposable
{
    private nint _problem;
    private readonly GCHandle[] _callbackHandles;
    private bool _disposed;

    public unsafe IpoptSolver(
        int n,
        ReadOnlySpan<double> xL,
        ReadOnlySpan<double> xU,
        int m,
        ReadOnlySpan<double> gL,
        ReadOnlySpan<double> gU,
        int jacobianNonZeros,
        int hessianNonZeros,
        EvalFCallback evalF,
        EvalGradFCallback evalGradF,
        EvalGCallback evalG,
        EvalJacGCallback evalJacG,
        EvalHCallback evalH)
    {
        if (xL.Length != n || xU.Length != n)
            throw new ArgumentException($"Variable bounds must have length {n}");
        if (m > 0 && (gL.Length != m || gU.Length != m))
            throw new ArgumentException($"Constraint bounds must have length {m}");

        _callbackHandles = new GCHandle[5];
        _callbackHandles[0] = GCHandle.Alloc(evalF);
        _callbackHandles[1] = GCHandle.Alloc(evalGradF);
        _callbackHandles[2] = GCHandle.Alloc(evalG);
        _callbackHandles[3] = GCHandle.Alloc(evalJacG);
        _callbackHandles[4] = GCHandle.Alloc(evalH);

        fixed (double* pXL = xL)
        fixed (double* pXU = xU)
        fixed (double* pGL = gL)
        fixed (double* pGU = gU)
        {
            _problem = Native.CreateIpoptProblem(
                n, pXL, pXU,
                m, pGL, pGU,
                jacobianNonZeros,
                hessianNonZeros,
                0, // C-style indexing (0-based)
                Marshal.GetFunctionPointerForDelegate(evalF),
                Marshal.GetFunctionPointerForDelegate(evalG),
                Marshal.GetFunctionPointerForDelegate(evalGradF),
                Marshal.GetFunctionPointerForDelegate(evalJacG),
                Marshal.GetFunctionPointerForDelegate(evalH));
        }

        if (_problem == 0)
            throw new InvalidOperationException("Failed to create IPOPT problem");
    }

    public bool SetOption(string name, string value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Native.AddIpoptStrOption(_problem, name, value);
    }

    public bool SetOption(string name, double value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Native.AddIpoptNumOption(_problem, name, value);
    }

    public bool SetOption(string name, int value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Native.AddIpoptIntOption(_problem, name, value);
    }

    public unsafe ApplicationReturnStatus Solve(
        Span<double> x,
        out double objectiveValue,
        Span<double> constraintValues = default,
        Span<double> constraintMultipliers = default,
        Span<double> lowerBoundMultipliers = default,
        Span<double> upperBoundMultipliers = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        double objVal = 0;

        fixed (double* pX = x)
        fixed (double* pG = constraintValues)
        fixed (double* pMultG = constraintMultipliers)
        fixed (double* pMultXL = lowerBoundMultipliers)
        fixed (double* pMultXU = upperBoundMultipliers)
        {
            var status = Native.IpoptSolve(_problem, pX, pG, &objVal, pMultG, pMultXL, pMultXU, 0);
            objectiveValue = objVal;
            return status;
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;

        if (_problem != 0)
        {
            Native.FreeIpoptProblem(_problem);
            _problem = 0;
        }

        foreach (var handle in _callbackHandles)
        {
            if (handle.IsAllocated)
                handle.Free();
        }

        GC.SuppressFinalize(this);
    }

    ~IpoptSolver()
    {
        Dispose();
    }
}
