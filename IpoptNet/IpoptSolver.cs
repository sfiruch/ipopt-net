using System.Runtime.InteropServices;

namespace IpoptNet;

public sealed record SolveStatistics(
    int IterationCount,
    double FinalObjectiveValue,
    double PrimalInfeasibility,
    double DualInfeasibility,
    double ComplementarityMeasure,
    int LineSearchTrials);

public sealed class IpoptSolver : IDisposable
{
    private nint _problem;
    private readonly GCHandle[] _callbackHandles;
    private bool _disposed;
    private int _lastIterationCount;
    private double _lastObjectiveValue;
    private double _lastPrimalInfeasibility;
    private double _lastDualInfeasibility;
    private double _lastComplementarity;
    private int _lastLineSearchTrials;

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

        // Set up intermediate callback to capture statistics
        var intermediateCallback = new IntermediateCallback(IntermediateCallbackHandler);

        _callbackHandles = new GCHandle[6];
        _callbackHandles[0] = GCHandle.Alloc(evalF);
        _callbackHandles[1] = GCHandle.Alloc(evalGradF);
        _callbackHandles[2] = GCHandle.Alloc(evalG);
        _callbackHandles[3] = GCHandle.Alloc(evalJacG);
        _callbackHandles[4] = GCHandle.Alloc(evalH);
        _callbackHandles[5] = GCHandle.Alloc(intermediateCallback);

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

        Native.SetIntermediateCallback(_problem, Marshal.GetFunctionPointerForDelegate(intermediateCallback));
    }

    private unsafe bool IntermediateCallbackHandler(
        AlgorithmMode algMode,
        int iterCount,
        double objValue,
        double infPr,
        double infDu,
        double mu,
        double dNorm,
        double regularizationSize,
        double alphaDu,
        double alphaPr,
        int lsTrials,
        nint userData)
    {
        _lastIterationCount = iterCount;
        _lastObjectiveValue = objValue;
        _lastPrimalInfeasibility = infPr;
        _lastDualInfeasibility = infDu;
        _lastComplementarity = mu;
        _lastLineSearchTrials = lsTrials;
        return true;
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
        out SolveStatistics statistics,
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
            statistics = new SolveStatistics(
                _lastIterationCount,
                _lastObjectiveValue,
                _lastPrimalInfeasibility,
                _lastDualInfeasibility,
                _lastComplementarity,
                _lastLineSearchTrials);
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
