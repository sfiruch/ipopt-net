using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class ExpressionTests
{
    private const double FiniteDiffDelta = 1e-6;
    private const double GradientTolerance = 1e-4;
    private const double HessianTolerance = 5e-3;

    [TestMethod]
    public void Gradient_Addition_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = x + y + 3.0;
        double[] point = [2, 3];

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Gradient_Multiplication_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = x * y;
        double[] point = [2, 3];

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Gradient_ComplexExpression_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // x^2 * y + sin(x) + exp(y)
        var expr = x * x * y + Expr.Sin(x) + Expr.Exp(y);
        double[] point = [1.5, 0.5];

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Gradient_Division_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = x / y;
        double[] point = [3, 2];

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Gradient_Power_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();

        var expr = Expr.Pow(x, 3);
        double[] point = [2];

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Gradient_Trigonometric_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();

        var sin = Expr.Sin(x);
        var cos = Expr.Cos(x);
        var tan = Expr.Tan(x);
        double[] point = [0.5];

        AssertGradientMatchesFiniteDifference(sin, point);
        AssertGradientMatchesFiniteDifference(cos, point);
        AssertGradientMatchesFiniteDifference(tan, point);
    }

    [TestMethod]
    public void Gradient_Log_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();

        var expr = Expr.Log(x);
        double[] point = [2];

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Hessian_Quadratic_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // x^2 + 2*x*y + y^2
        var expr = x * x + 2 * x * y + y * y;
        double[] point = [1, 2];

        AssertHessianMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Hessian_HS071Objective_MatchesFiniteDifference()
    {
        var model = new Model();
        var x1 = model.AddVariable();
        var x2 = model.AddVariable();
        var x3 = model.AddVariable();
        var x4 = model.AddVariable();

        // x1*x4*(x1+x2+x3) + x3
        var expr = x1 * x4 * (x1 + x2 + x3) + x3;
        double[] point = [1, 5, 5, 1];

        AssertHessianMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Hessian_Rosenbrock_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // (1-x)^2 + 100*(y-x^2)^2
        var expr = Expr.Pow(1 - x, 2) + 100 * Expr.Pow(y - x * x, 2);
        double[] point = [0.5, 0.5];

        AssertHessianMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Hessian_Trigonometric_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();

        var sin = Expr.Sin(x);
        var cos = Expr.Cos(x);
        double[] point = [1.0];

        AssertHessianMatchesFiniteDifference(sin, point);
        AssertHessianMatchesFiniteDifference(cos, point);
    }

    [TestMethod]
    public void Hessian_Exp_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();

        var expr = Expr.Exp(x);
        double[] point = [1.0];

        AssertHessianMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Hessian_Division_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = x / y;
        double[] point = [3, 2];

        AssertHessianMatchesFiniteDifference(expr, point);
    }

    private static void AssertGradientMatchesFiniteDifference(Expr expr, double[] point)
    {
        var n = point.Length;
        var adGrad = new double[n];
        expr.AccumulateGradient(point, adGrad, 1.0);

        var fdGrad = ComputeFiniteDifferenceGradient(expr, point);

        for (int i = 0; i < n; i++)
            Assert.AreEqual(fdGrad[i], adGrad[i], GradientTolerance, $"Gradient mismatch at index {i}");
    }

    private static void AssertHessianMatchesFiniteDifference(Expr expr, double[] point)
    {
        var n = point.Length;
        var grad = new double[n];
        var hess = new HessianAccumulator(n);
        expr.AccumulateHessian(point, grad, hess, 1.0);

        var fdHess = ComputeFiniteDifferenceHessian(expr, point);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                hess.Entries.TryGetValue((i, j), out var adValue);
                Assert.AreEqual(fdHess[i, j], adValue, HessianTolerance,
                    $"Hessian mismatch at ({i},{j}): FD={fdHess[i, j]}, AD={adValue}");
            }
        }
    }

    private static double[] ComputeFiniteDifferenceGradient(Expr expr, double[] point)
    {
        var n = point.Length;
        var grad = new double[n];
        var xPlus = (double[])point.Clone();
        var xMinus = (double[])point.Clone();

        for (int i = 0; i < n; i++)
        {
            xPlus[i] = point[i] + FiniteDiffDelta;
            xMinus[i] = point[i] - FiniteDiffDelta;
            var fPlus = expr.Evaluate(xPlus);
            var fMinus = expr.Evaluate(xMinus);
            grad[i] = (fPlus - fMinus) / (2 * FiniteDiffDelta);
            xPlus[i] = point[i];
            xMinus[i] = point[i];
        }

        return grad;
    }

    private static double[,] ComputeFiniteDifferenceHessian(Expr expr, double[] point)
    {
        var n = point.Length;
        var hess = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                // Use central difference for second derivative
                var xpp = (double[])point.Clone();
                var xpm = (double[])point.Clone();
                var xmp = (double[])point.Clone();
                var xmm = (double[])point.Clone();

                xpp[i] += FiniteDiffDelta;
                xpp[j] += FiniteDiffDelta;
                xpm[i] += FiniteDiffDelta;
                xpm[j] -= FiniteDiffDelta;
                xmp[i] -= FiniteDiffDelta;
                xmp[j] += FiniteDiffDelta;
                xmm[i] -= FiniteDiffDelta;
                xmm[j] -= FiniteDiffDelta;

                var fpp = expr.Evaluate(xpp);
                var fpm = expr.Evaluate(xpm);
                var fmp = expr.Evaluate(xmp);
                var fmm = expr.Evaluate(xmm);

                hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * FiniteDiffDelta * FiniteDiffDelta);
                hess[j, i] = hess[i, j];
            }
        }

        return hess;
    }

    [TestMethod]
    public void IncrementalBuild_AdditionWithZero_ProducesCorrectResult()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();

        // Build expression incrementally: sum = x + y + z
        Expr sum = 0;
        sum += x;
        sum += y;
        sum += z;

        double[] point = [1, 2, 3];
        var result = sum.Evaluate(point);
        Assert.AreEqual(6.0, result, 1e-10);

        AssertGradientMatchesFiniteDifference(sum, point);
    }

    [TestMethod]
    public void IncrementalBuild_SubtractionFromZero_ProducesCorrectResult()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Build expression incrementally: expr = -x - y
        Expr expr = 0;
        expr -= x;
        expr -= y;

        double[] point = [2, 3];
        var result = expr.Evaluate(point);
        Assert.AreEqual(-5.0, result, 1e-10);

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void IncrementalBuild_MultiplicationWithOne_ProducesCorrectResult()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();

        // Build expression incrementally: product = x * y * z
        Expr product = 1;
        product *= x;
        product *= y;
        product *= z;

        double[] point = [2, 3, 4];
        var result = product.Evaluate(point);
        Assert.AreEqual(24.0, result, 1e-10);

        AssertGradientMatchesFiniteDifference(product, point);
    }

    [TestMethod]
    public void IncrementalBuild_DivisionFromOne_ProducesCorrectResult()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Build expression incrementally: expr = 1 / (x * y)
        Expr expr = 1;
        expr /= x;
        expr /= y;

        double[] point = [2, 5];
        var result = expr.Evaluate(point);
        Assert.AreEqual(0.1, result, 1e-10);

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void IncrementalBuild_MixedOperations_ProducesCorrectResult()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();

        // Build complex expression incrementally
        Expr expr = 0;
        expr += x * x;      // x^2
        expr += 2 * x * y;  // + 2xy
        expr -= z;          // - z
        expr *= 2;          // * 2

        double[] point = [1, 2, 3];
        // (1^2 + 2*1*2 - 3) * 2 = (1 + 4 - 3) * 2 = 4
        var result = expr.Evaluate(point);
        Assert.AreEqual(4.0, result, 1e-10);

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void IncrementalBuild_WithDoubles_ProducesCorrectResult()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Build expression incrementally with constants
        Expr expr = 0;
        expr += x;
        expr += 5.0;
        expr *= 2.0;

        double[] point = [3];
        // (3 + 5) * 2 = 16
        var result = expr.Evaluate(point);
        Assert.AreEqual(16.0, result, 1e-10);

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void IncrementalBuild_LargeSum_ProducesCorrectResult()
    {
        var model = new Model();
        var variables = new Variable[10];
        for (int i = 0; i < 10; i++)
            variables[i] = model.AddVariable();

        // Build a large sum incrementally using compound assignment
        Expr sum = 0;
        for (int i = 0; i < 10; i++)
            sum += variables[i];

        var point = new double[10];
        for (int i = 0; i < 10; i++)
            point[i] = i + 1; // 1, 2, 3, ..., 10

        var result = sum.Evaluate(point);
        Assert.AreEqual(55.0, result, 1e-10); // Sum of 1 to 10

        AssertGradientMatchesFiniteDifference(sum, point);
    }

    [TestMethod]
    public void ImplicitConversion_IntToExpr_CreatesConstant()
    {
        Expr zero = 0;
        Expr one = 1;
        Expr five = 5;

        Assert.IsInstanceOfType(zero, typeof(Constant));
        Assert.IsInstanceOfType(one, typeof(Constant));
        Assert.IsInstanceOfType(five, typeof(Constant));

        Assert.AreEqual(0.0, ((Constant)zero).Value);
        Assert.AreEqual(1.0, ((Constant)one).Value);
        Assert.AreEqual(5.0, ((Constant)five).Value);
    }
}
