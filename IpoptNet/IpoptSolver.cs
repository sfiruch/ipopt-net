using System.Runtime.CompilerServices;
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
    private GCHandle _selfHandle;
    private bool _disposed;

    // Store callback delegates as fields to keep them alive
    private readonly EvalFCallback _evalF;
    private readonly EvalGradFCallback _evalGradF;
    private readonly EvalGCallback _evalG;
    private readonly EvalJacGCallback _evalJacG;
    private readonly EvalHCallback _evalH;

    // Statistics captured from intermediate callback
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

        // Store callbacks as fields to keep them alive
        _evalF = evalF;
        _evalGradF = evalGradF;
        _evalG = evalG;
        _evalJacG = evalJacG;
        _evalH = evalH;

        // Allocate GCHandle for this instance to pass as userData
        _selfHandle = GCHandle.Alloc(this);

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
                (nint)(delegate*unmanaged[Cdecl]<int, double*, int, double*, nint, int>)&EvalFStatic,
                (nint)(delegate*unmanaged[Cdecl]<int, double*, int, int, double*, nint, int>)&EvalGStatic,
                (nint)(delegate*unmanaged[Cdecl]<int, double*, int, double*, nint, int>)&EvalGradFStatic,
                (nint)(delegate*unmanaged[Cdecl]<int, double*, int, int, int, int*, int*, double*, nint, int>)&EvalJacGStatic,
                (nint)(delegate*unmanaged[Cdecl]<int, double*, int, double, int, double*, int, int, int*, int*, double*, nint, int>)&EvalHStatic);
        }

        if (_problem == 0)
            throw new InvalidOperationException("Failed to create IPOPT problem");

        Native.SetIntermediateCallback(_problem,
            (nint)(delegate*unmanaged[Cdecl]<AlgorithmMode, int, double, double, double, double, double, double, double, double, int, nint, int>)&IntermediateCallbackStatic);
    }

    // Static callback methods with UnmanagedCallersOnly
    // Note: Use int instead of bool for blittable types (0 = false, non-zero = true)
    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe int EvalFStatic(int n, double* x, int newX, double* objValue, nint userData)
    {
        var instance = (IpoptSolver)GCHandle.FromIntPtr(userData).Target!;
        return instance._evalF(n, x, newX != 0, objValue, userData) ? 1 : 0;
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe int EvalGradFStatic(int n, double* x, int newX, double* gradF, nint userData)
    {
        var instance = (IpoptSolver)GCHandle.FromIntPtr(userData).Target!;
        return instance._evalGradF(n, x, newX != 0, gradF, userData) ? 1 : 0;
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe int EvalGStatic(int n, double* x, int newX, int m, double* g, nint userData)
    {
        var instance = (IpoptSolver)GCHandle.FromIntPtr(userData).Target!;
        return instance._evalG(n, x, newX != 0, m, g, userData) ? 1 : 0;
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe int EvalJacGStatic(int n, double* x, int newX, int m, int neleJac, int* iRow, int* jCol, double* values, nint userData)
    {
        var instance = (IpoptSolver)GCHandle.FromIntPtr(userData).Target!;
        return instance._evalJacG(n, x, newX != 0, m, neleJac, iRow, jCol, values, userData) ? 1 : 0;
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe int EvalHStatic(int n, double* x, int newX, double objFactor, int m, double* lambda, int newLambda, int neleHess, int* iRow, int* jCol, double* values, nint userData)
    {
        var instance = (IpoptSolver)GCHandle.FromIntPtr(userData).Target!;
        return instance._evalH(n, x, newX != 0, objFactor, m, lambda, newLambda != 0, neleHess, iRow, jCol, values, userData) ? 1 : 0;
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe int IntermediateCallbackStatic(
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
        var instance = (IpoptSolver)GCHandle.FromIntPtr(userData).Target!;
        instance._lastIterationCount = iterCount;
        instance._lastObjectiveValue = objValue;
        instance._lastPrimalInfeasibility = infPr;
        instance._lastDualInfeasibility = infDu;
        instance._lastComplementarity = mu;
        instance._lastLineSearchTrials = lsTrials;
        return 1;
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
            var status = Native.IpoptSolve(_problem, pX, pG, &objVal, pMultG, pMultXL, pMultXU, GCHandle.ToIntPtr(_selfHandle));
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

        if (_selfHandle.IsAllocated)
            _selfHandle.Free();

        GC.SuppressFinalize(this);
    }

    ~IpoptSolver()
    {
        Dispose();
    }
}
