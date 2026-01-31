using System.Runtime.InteropServices;

namespace IpoptNet;

public enum ApplicationReturnStatus
{
    SolveSucceeded = 0,
    SolvedToAcceptableLevel = 1,
    InfeasibleProblemDetected = 2,
    SearchDirectionBecomesTooSmall = 3,
    DivergingIterates = 4,
    UserRequestedStop = 5,
    FeasiblePointFound = 6,
    MaximumIterationsExceeded = -1,
    RestorationFailed = -2,
    ErrorInStepComputation = -3,
    MaximumCpuTimeExceeded = -4,
    MaximumWallTimeExceeded = -5,
    NotEnoughDegreesOfFreedom = -10,
    InvalidProblemDefinition = -11,
    InvalidOption = -12,
    InvalidNumberDetected = -13,
    UnrecoverableException = -100,
    NonIpoptExceptionThrown = -101,
    InsufficientMemory = -102,
    InternalError = -199
}

public enum AlgorithmMode
{
    RegularMode = 0,
    RestorationPhaseMode = 1
}

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public unsafe delegate bool EvalFCallback(
    int n,
    double* x,
    bool newX,
    double* objValue,
    nint userData);

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public unsafe delegate bool EvalGradFCallback(
    int n,
    double* x,
    bool newX,
    double* gradF,
    nint userData);

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public unsafe delegate bool EvalGCallback(
    int n,
    double* x,
    bool newX,
    int m,
    double* g,
    nint userData);

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public unsafe delegate bool EvalJacGCallback(
    int n,
    double* x,
    bool newX,
    int m,
    int nele_jac,
    int* iRow,
    int* jCol,
    double* values,
    nint userData);

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public unsafe delegate bool EvalHCallback(
    int n,
    double* x,
    bool newX,
    double objFactor,
    int m,
    double* lambda,
    bool newLambda,
    int nele_hess,
    int* iRow,
    int* jCol,
    double* values,
    nint userData);

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public unsafe delegate bool IntermediateCallback(
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
    nint userData);

internal static partial class Native
{
    private const string LibraryName = "ipopt-3";

    [LibraryImport(LibraryName, EntryPoint = "CreateIpoptProblem")]
    public static unsafe partial nint CreateIpoptProblem(
        int n,
        double* xL,
        double* xU,
        int m,
        double* gL,
        double* gU,
        int neleJac,
        int neleHess,
        int indexStyle,
        nint evalF,
        nint evalG,
        nint evalGradF,
        nint evalJacG,
        nint evalH);

    [LibraryImport(LibraryName, EntryPoint = "FreeIpoptProblem")]
    public static partial void FreeIpoptProblem(nint problem);

    [LibraryImport(LibraryName, EntryPoint = "AddIpoptStrOption", StringMarshalling = StringMarshalling.Utf8)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool AddIpoptStrOption(nint problem, string keyword, string value);

    [LibraryImport(LibraryName, EntryPoint = "AddIpoptNumOption", StringMarshalling = StringMarshalling.Utf8)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool AddIpoptNumOption(nint problem, string keyword, double value);

    [LibraryImport(LibraryName, EntryPoint = "AddIpoptIntOption", StringMarshalling = StringMarshalling.Utf8)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool AddIpoptIntOption(nint problem, string keyword, int value);

    [LibraryImport(LibraryName, EntryPoint = "IpoptSolve")]
    public static unsafe partial ApplicationReturnStatus IpoptSolve(
        nint problem,
        double* x,
        double* g,
        double* objVal,
        double* multG,
        double* multXL,
        double* multXU,
        nint userData);

    [LibraryImport(LibraryName, EntryPoint = "SetIpoptProblemScaling")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static unsafe partial bool SetIpoptProblemScaling(
        nint problem,
        double objScaling,
        double* xScaling,
        double* gScaling);

    [LibraryImport(LibraryName, EntryPoint = "SetIntermediateCallback")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool SetIntermediateCallback(nint problem, nint callback);

    [LibraryImport(LibraryName, EntryPoint = "GetIpoptCurrentIterate")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static unsafe partial bool GetIpoptCurrentIterate(
        nint problem,
        [MarshalAs(UnmanagedType.Bool)] bool scaled,
        int n,
        double* x,
        double* zL,
        double* zU,
        int m,
        double* g,
        double* lambda);

    [LibraryImport(LibraryName, EntryPoint = "GetIpoptCurrentViolations")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static unsafe partial bool GetIpoptCurrentViolations(
        nint problem,
        [MarshalAs(UnmanagedType.Bool)] bool scaled,
        int n,
        double* xLViolation,
        double* xUViolation,
        double* complXL,
        double* complXU,
        double* gradLagX,
        int m,
        double* gViolation,
        double* complG);
}
