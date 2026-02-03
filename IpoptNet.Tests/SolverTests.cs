using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class SolverTests
{
    /// <summary>
    /// Tests the classic HS071 problem from the Hock-Schittkowski test suite.
    /// minimize: x1*x4*(x1+x2+x3) + x3
    /// subject to: x1*x2*x3*x4 >= 25
    ///             x1^2 + x2^2 + x3^2 + x4^2 = 40
    ///             1 <= x1,x2,x3,x4 <= 5
    /// Starting point: (1,5,5,1)
    /// Optimal solution: x* ≈ (1.0, 4.743, 3.821, 1.379)
    /// Optimal value: f(x*) ≈ 17.014
    /// </summary>
    [TestMethod]
    public unsafe void HS071_Converges()
    {
        const int n = 4;
        const int m = 2;
        const int jacobianNonZeros = 8; // All 4 vars appear in both constraints
        const int hessianNonZeros = 10; // Lower triangular 4x4

        double[] xL = [1, 1, 1, 1];
        double[] xU = [5, 5, 5, 5];
        double[] gL = [25, 40];
        double[] gU = [double.PositiveInfinity, 40];

        // Callback: evaluate objective f(x) = x1*x4*(x1+x2+x3) + x3
        EvalFCallback evalF = (int nn, double* x, bool newX, double* objValue, nint userData) =>
        {
            *objValue = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
            return true;
        };

        // Callback: evaluate gradient of f
        EvalGradFCallback evalGradF = (int nn, double* x, bool newX, double* gradF, nint userData) =>
        {
            gradF[0] = x[3] * (2 * x[0] + x[1] + x[2]);
            gradF[1] = x[0] * x[3];
            gradF[2] = x[0] * x[3] + 1;
            gradF[3] = x[0] * (x[0] + x[1] + x[2]);
            return true;
        };

        // Callback: evaluate constraints
        // g1 = x1*x2*x3*x4
        // g2 = x1^2 + x2^2 + x3^2 + x4^2
        EvalGCallback evalG = (int nn, double* x, bool newX, int mm, double* g, nint userData) =>
        {
            g[0] = x[0] * x[1] * x[2] * x[3];
            g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
            return true;
        };

        // Jacobian structure (row-major):
        // Row 0 (g1): cols 0,1,2,3
        // Row 1 (g2): cols 0,1,2,3
        int[] jacRows = [0, 0, 0, 0, 1, 1, 1, 1];
        int[] jacCols = [0, 1, 2, 3, 0, 1, 2, 3];

        EvalJacGCallback evalJacG = (int nn, double* x, bool newX, int mm, int neleJac, int* iRow, int* jCol, double* values, nint userData) =>
        {
            if (values == null)
            {
                for (int i = 0; i < jacRows.Length; i++)
                {
                    iRow[i] = jacRows[i];
                    jCol[i] = jacCols[i];
                }
            }
            else
            {
                // g1 = x1*x2*x3*x4
                values[0] = x[1] * x[2] * x[3];
                values[1] = x[0] * x[2] * x[3];
                values[2] = x[0] * x[1] * x[3];
                values[3] = x[0] * x[1] * x[2];
                // g2 = x1^2 + x2^2 + x3^2 + x4^2
                values[4] = 2 * x[0];
                values[5] = 2 * x[1];
                values[6] = 2 * x[2];
                values[7] = 2 * x[3];
            }
            return true;
        };

        // Hessian structure (lower triangular):
        // (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)
        int[] hessRows = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3];
        int[] hessCols = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3];

        EvalHCallback evalH = (int nn, double* x, bool newX, double objFactor, int mm, double* lambda, bool newLambda,
                              int neleHess, int* iRow, int* jCol, double* values, nint userData) =>
        {
            if (values == null)
            {
                for (int i = 0; i < hessRows.Length; i++)
                {
                    iRow[i] = hessRows[i];
                    jCol[i] = hessCols[i];
                }
            }
            else
            {
                // Initialize all to zero
                for (int i = 0; i < neleHess; i++)
                    values[i] = 0;

                // Objective Hessian contributions (f = x1*x4*(x1+x2+x3) + x3)
                // d2f/dx1^2 = 2*x4
                values[0] += objFactor * 2 * x[3];
                // d2f/dx1dx2 = x4
                values[1] += objFactor * x[3];
                // d2f/dx1dx3 = x4
                values[3] += objFactor * x[3];
                // d2f/dx1dx4 = 2*x1 + x2 + x3
                values[6] += objFactor * (2 * x[0] + x[1] + x[2]);
                // d2f/dx2dx4 = x1
                values[7] += objFactor * x[0];
                // d2f/dx3dx4 = x1
                values[8] += objFactor * x[0];

                // Constraint 1 Hessian (g1 = x1*x2*x3*x4)
                // d2g1/dx1dx2 = x3*x4
                values[1] += lambda[0] * x[2] * x[3];
                // d2g1/dx1dx3 = x2*x4
                values[3] += lambda[0] * x[1] * x[3];
                // d2g1/dx2dx3 = x1*x4
                values[4] += lambda[0] * x[0] * x[3];
                // d2g1/dx1dx4 = x2*x3
                values[6] += lambda[0] * x[1] * x[2];
                // d2g1/dx2dx4 = x1*x3
                values[7] += lambda[0] * x[0] * x[2];
                // d2g1/dx3dx4 = x1*x2
                values[8] += lambda[0] * x[0] * x[1];

                // Constraint 2 Hessian (g2 = x1^2 + x2^2 + x3^2 + x4^2)
                values[0] += lambda[1] * 2;  // d2g2/dx1^2
                values[2] += lambda[1] * 2;  // d2g2/dx2^2
                values[5] += lambda[1] * 2;  // d2g2/dx3^2
                values[9] += lambda[1] * 2;  // d2g2/dx4^2
            }
            return true;
        };

        using var solver = new IpoptSolver(n, xL, xU, m, gL, gU, jacobianNonZeros, hessianNonZeros,
            evalF, evalGradF, evalG, evalJacG, evalH);

        solver.SetOption("derivative_test", "second-order");
        solver.SetOption("print_level", 0);

        double[] x = [1, 5, 5, 1];
        var status = solver.Solve(x, out var objValue, out var statistics);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, status);
        Assert.AreEqual(17.014, objValue, 0.01);
        Assert.IsTrue(statistics.IterationCount > 0);
        Assert.AreEqual(1.0, x[0], 0.01);
        Assert.AreEqual(4.743, x[1], 0.01);
        Assert.AreEqual(3.821, x[2], 0.01);
        Assert.AreEqual(1.379, x[3], 0.01);
    }
}
