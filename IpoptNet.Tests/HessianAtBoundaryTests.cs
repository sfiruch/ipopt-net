using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

/// <summary>
/// Regression tests for Hessian correctness at points where one or more sub-expressions
/// evaluate to zero. The failure mode this catches: computing a derivative coefficient via
/// division by an evaluated sub-expression that is mathematically smooth but happens to be
/// zero at the query point (e.g. ProductNode using totalProduct/factor[i] to get "product
/// except i"). Such code produces 0/0 = NaN in the Hessian, which poisons Pardiso and can
/// crash the process many solves later when the NaN propagates through internal caches.
///
/// The existing Hessian tests (Hessian_TwoFactorProduct_*, Hessian_ThreeFactorProduct_*)
/// use non-zero values (e.g. [2, 3, 5]) and therefore cannot detect this class of bug.
/// </summary>
[TestClass]
public class HessianAtBoundaryTests
{
    private const double FiniteDiffDelta = 1e-6;
    private const double HessianTolerance = 5e-3;

    [TestMethod]
    public void Hessian_Product_xy_xZero_IsFiniteAndCorrect()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        AssertHessianFiniteAndMatchesFiniteDifference(x * y, [0.0, 5.0]);
    }

    [TestMethod]
    public void Hessian_Product_xy_yZero_IsFiniteAndCorrect()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        AssertHessianFiniteAndMatchesFiniteDifference(x * y, [5.0, 0.0]);
    }

    [TestMethod]
    public void Hessian_Product_xy_BothZero_IsFiniteAndCorrect()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        AssertHessianFiniteAndMatchesFiniteDifference(x * y, [0.0, 0.0]);
    }

    [TestMethod]
    public void Hessian_Product_xyz_OneFactorZero_IsFiniteAndCorrect()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();
        AssertHessianFiniteAndMatchesFiniteDifference(x * y * z, [0.0, 3.0, 5.0]);
    }

    [TestMethod]
    public void Hessian_Product_xyz_TwoFactorsZero_IsFiniteAndCorrect()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();
        AssertHessianFiniteAndMatchesFiniteDifference(x * y * z, [0.0, 0.0, 5.0]);
    }

    [TestMethod]
    public void Hessian_Product_xyz_AllZero_IsFiniteAndCorrect()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();
        AssertHessianFiniteAndMatchesFiniteDifference(x * y * z, [0.0, 0.0, 0.0]);
    }

    /// <summary>
    /// Mirrors the ZoneGenius failure mode: a bilinear product where one side is a sum of
    /// variable*constant terms (e.g. a ventilation normalisation q_vent_norm = Σ coef_i·dc_i)
    /// and can legitimately evaluate to zero when all dc_i are zero at some inner iteration.
    /// </summary>
    [TestMethod]
    public void Hessian_BilinearProduct_InnerSumZero_IsFiniteAndCorrect()
    {
        var model = new Model();
        var x = model.AddVariable();
        var dc1 = model.AddVariable();
        var dc2 = model.AddVariable();
        // x * (2·dc1 + 3·dc2): bilinear; the inner sum hits zero when both dc's are zero.
        AssertHessianFiniteAndMatchesFiniteDifference(x * (2.0 * dc1 + 3.0 * dc2), [4.0, 0.0, 0.0]);
    }

    [TestMethod]
    public void Hessian_BilinearProduct_OuterZero_IsFiniteAndCorrect()
    {
        var model = new Model();
        var x = model.AddVariable();
        var dc1 = model.AddVariable();
        var dc2 = model.AddVariable();
        // Same structure, but now the outer factor is zero.
        AssertHessianFiniteAndMatchesFiniteDifference(x * (2.0 * dc1 + 3.0 * dc2), [0.0, 1.5, 2.0]);
    }

    /// <summary>
    /// PowerOp is already guarded (it zeroes out non-finite second-derivative coefficients),
    /// so x² at x=0 must be finite. This is a regression lock-in, not a bug detection test.
    /// </summary>
    [TestMethod]
    public void Hessian_PowerSquared_AtZero_IsFinite()
    {
        var model = new Model();
        var x = model.AddVariable();
        AssertHessianFiniteAndMatchesFiniteDifference(Expr.Pow(x, 2), [0.0]);
    }

    private static void AssertHessianFiniteAndMatchesFiniteDifference(Expr expr, double[] point)
    {
        expr.Prepare();
        var n = point.Length;

        var entries = new HashSet<(int row, int col)>();
        expr.CollectHessianSparsity(entries);
        var sorted = entries.OrderBy(e => e.row).ThenBy(e => e.col).ToArray();
        var rows = sorted.Select(e => e.row).ToArray();
        var cols = sorted.Select(e => e.col).ToArray();

        var hess = new HessianAccumulator(n, rows, cols);
        expr.AccumulateHessian(point, hess, 1.0);

        // Fail fast on NaN/Inf: this is the primary symptom the tests exist to catch.
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                var adValue = hess.Get(i, j);
                Assert.IsTrue(double.IsFinite(adValue),
                    $"Hessian[{i},{j}] is not finite ({adValue}) at point [{string.Join(", ", point)}] for expression producing NaN/Inf.");
            }

        var fdHess = ComputeFiniteDifferenceHessian(expr, point);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                Assert.AreEqual(fdHess[i, j], hess.Get(i, j), HessianTolerance,
                    $"Hessian mismatch at ({i},{j}): FD={fdHess[i, j]}, AD={hess.Get(i, j)}, point=[{string.Join(", ", point)}]");
    }

    private static double[,] ComputeFiniteDifferenceHessian(Expr expr, double[] point)
    {
        var n = point.Length;
        var fdHess = new double[n, n];
        var xPP = (double[])point.Clone();
        var xPM = (double[])point.Clone();
        var xMP = (double[])point.Clone();
        var xMM = (double[])point.Clone();
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                xPP[i] = point[i] + FiniteDiffDelta; xPP[j] += FiniteDiffDelta;
                xPM[i] = point[i] + FiniteDiffDelta; xPM[j] -= FiniteDiffDelta;
                xMP[i] = point[i] - FiniteDiffDelta; xMP[j] += FiniteDiffDelta;
                xMM[i] = point[i] - FiniteDiffDelta; xMM[j] -= FiniteDiffDelta;
                if (i == j)
                {
                    // Diagonal: second derivative via central differences on the same axis
                    xPP[j] = point[j];
                    xPP[i] = point[i] + FiniteDiffDelta;
                    xMM[j] = point[j];
                    xMM[i] = point[i] - FiniteDiffDelta;
                    var fPlus = expr.Evaluate(xPP);
                    var fMinus = expr.Evaluate(xMM);
                    var fCenter = expr.Evaluate(point);
                    fdHess[i, j] = (fPlus - 2 * fCenter + fMinus) / (FiniteDiffDelta * FiniteDiffDelta);
                }
                else
                {
                    fdHess[i, j] = (expr.Evaluate(xPP) - expr.Evaluate(xPM) - expr.Evaluate(xMP) + expr.Evaluate(xMM))
                                   / (4 * FiniteDiffDelta * FiniteDiffDelta);
                    fdHess[j, i] = fdHess[i, j];
                }
                xPP[i] = point[i]; xPP[j] = point[j];
                xPM[i] = point[i]; xPM[j] = point[j];
                xMP[i] = point[i]; xMP[j] = point[j];
                xMM[i] = point[i]; xMM[j] = point[j];
            }
        return fdHess;
    }
}
