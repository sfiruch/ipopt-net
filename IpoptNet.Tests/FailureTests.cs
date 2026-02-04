using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class FailureTests
{
    [TestMethod]
    public unsafe void EvalF_ReturnsFalse_AbortsSolve()
    {
        const int n = 1;
        const int m = 0;
        double[] xL = [-10];
        double[] xU = [10];
        double[] gL = [];
        double[] gU = [];

        EvalFCallback evalF = (int nn, double* x, bool newX, double* objValue, nint userData) =>
        {
            if (x[0] < 0) return false;
            *objValue = x[0] * x[0];
            return true;
        };

        EvalGradFCallback evalGradF = (int nn, double* x, bool newX, double* gradF, nint userData) =>
        {
            gradF[0] = 2 * x[0];
            return true;
        };

        EvalGCallback evalG = (int nn, double* x, bool newX, int mm, double* g, nint userData) => true;
        EvalJacGCallback evalJacG = (int nn, double* x, bool newX, int mm, int neleJac, int* iRow, int* jCol, double* values, nint userData) => true;
        EvalHCallback evalH = (int nn, double* x, bool newX, double objFactor, int mm, double* lambda, bool newLambda,
                              int neleHess, int* iRow, int* jCol, double* values, nint userData) =>
        {
            if (values != null) values[0] = objFactor * 2;
            else { iRow[0] = 0; jCol[0] = 0; }
            return true;
        };

        using var solver = new IpoptSolver(n, xL, xU, m, gL, gU, 0, 1,
            evalF, evalGradF, evalG, evalJacG, evalH);
        
        solver.SetOption("print_level", 0);

        double[] x = [-1.0]; // Starting at -1 should trigger false in evalF
        var status = solver.Solve(x, out var objValue, out var statistics);

        // Expected status when callback returns false at the initial point
        Assert.AreEqual(ApplicationReturnStatus.InvalidNumberDetected, status);
    }

    [TestMethod]
    public unsafe void EvalF_ReturnsFalse_Recoverable()
    {
        const int n = 1;
        const int m = 0;
        double[] xL = [-10];
        double[] xU = [10];
        double[] gL = [];
        double[] gU = [];

        // Minimize (x-0.5)^2
        // Starting at x=2.
        // If x < 1, return false.
        EvalFCallback evalF = (int nn, double* x, bool newX, double* objValue, nint userData) =>
        {
            if (x[0] < 1.0) return false;
            *objValue = (x[0] - 0.5) * (x[0] - 0.5);
            return true;
        };

        EvalGradFCallback evalGradF = (int nn, double* x, bool newX, double* gradF, nint userData) =>
        {
            if (x[0] < 1.0) return false;
            gradF[0] = 2 * (x[0] - 0.5);
            return true;
        };

        EvalGCallback evalG = (int nn, double* x, bool newX, int mm, double* g, nint userData) => true;
        EvalJacGCallback evalJacG = (int nn, double* x, bool newX, int mm, int neleJac, int* iRow, int* jCol, double* values, nint userData) => true;
        EvalHCallback evalH = (int nn, double* x, bool newX, double objFactor, int mm, double* lambda, bool newLambda,
                              int neleHess, int* iRow, int* jCol, double* values, nint userData) =>
        {
            if (values != null) values[0] = objFactor * 2;
            else { iRow[0] = 0; jCol[0] = 0; }
            return true;
        };

        using var solver = new IpoptSolver(n, xL, xU, m, gL, gU, 0, 1,
            evalF, evalGradF, evalG, evalJacG, evalH);
        
        solver.SetOption("print_level", 0);

        double[] x = [2.0];
        var status = solver.Solve(x, out var objValue, out var statistics);

        // It should stay in the x >= 1 region or terminate if it can't.
        // If it backtracks correctly, it shouldn't crash.
        Assert.AreNotEqual(ApplicationReturnStatus.InternalError, status);
    }
}
