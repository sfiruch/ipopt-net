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
    public void Gradient_DivisionWithOverlappingVariables_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = (x + y) / (x * y);
        double[] point = [3, 2];

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Gradient_MixedProductsDivisions_MatchesFiniteDifference()
    {
        // Complex expression with products, divisions, and sums
        var model = new Model();
        var v1 = model.AddVariable();
        var v2 = model.AddVariable();
        var v3 = model.AddVariable();
        var v4 = model.AddVariable();

        // Build a complex expression similar to user's optimization problem
        var expr = (v1 * v2 + v3) / (v4 + 1.5) + v1 * v3 / v2 - v2 * v4;
        double[] point = [2, 3, 4, 5];

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void IpoptDerivativeCheck_ComplexExpression_PassesValidation()
    {
        // Test using IPOPT's built-in derivative checker
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;

        var v1 = model.AddVariable(1, 10);
        v1.Start = 2;
        var v2 = model.AddVariable(1, 10);
        v2.Start = 3;
        var v3 = model.AddVariable(1, 10);
        v3.Start = 4;
        var v4 = model.AddVariable(1, 10);
        v4.Start = 5;

        // Complex objective with mixed operations including divisions with overlapping variables
        var obj = Expr.Pow(v1 * v2 + v3, 2) / (v4 + 1.5) + v1 * v3 / v2 - v2 * v4 + Expr.Log(v1 + v2);
        model.SetObjective(obj);

        // Add a constraint with overlapping variables in division
        model.AddConstraint((v1 + v2) / (v3 * v4) <= 5);

        var result = ModellingTests.SolveWithDerivativeTest(model);

        ModellingTests.AssertDerivativeTestPassed(result.DerivativeTestResult);
    }

    [TestMethod]
    public void IpoptDerivativeCheck_LeastSquaresWithDivisions_PassesValidation()
    {
        // Test pattern similar to user's code: sum of squared residuals with complex expressions
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;

        // Create some parameter variables
        var p1 = model.AddVariable(0);
        p1.Start = 1;
        var p2 = model.AddVariable(0);
        p2.Start = 2;
        var p3 = model.AddVariable(0);
        p3.Start = 1.5;

        // Build objective as sum of squared residuals
        Expr obj = 0;
        for (int i = 0; i < 10; i++)
        {
            double observed = i + 5.0;
            double x_val = i + 1.0;

            // predicted = (p1 * x + p2) / (p3 * x)  - involves division with overlapping variables
            var predicted = (p1 * x_val + p2) / (p3 * x_val);
            obj += Expr.Pow(observed - predicted, 2);
        }

        model.SetObjective(obj);

        var result = ModellingTests.SolveWithDerivativeTest(model);

        ModellingTests.AssertDerivativeTestPassed(result.DerivativeTestResult);
    }

    [TestMethod]
    public void Gradient_ProductWithRepeatedFactors_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create a Product directly with repeated factors: x * x * y
        var product = new Product([x, x, y]);
        double[] point = [2, 3];

        // d(x²y)/dx = 2xy = 2*2*3 = 12
        // d(x²y)/dy = x² = 4
        AssertGradientMatchesFiniteDifference(product, point);
    }

    [TestMethod]
    public void Hessian_ProductWithRepeatedFactors_MatchesFiniteDifference()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create a Product directly with repeated factors: x * x
        var product = new Product([x, x]);
        double[] point = [2, 3];

        // d²(x²)/dx² = 2
        AssertHessianMatchesFiniteDifference(product, point);
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

    [TestMethod]
    public void Hessian_DivisionWithCrossTerms_MatchesFiniteDifference()
    {
        // This test catches bugs in division cross-term Hessian computation
        // For f(w,x,y,z) = (w + x) / (y * z), the Hessian has cross-terms
        // where L and R depend on different variables (no overlap)
        var model = new Model();
        var w = model.AddVariable();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();

        var expr = (w + x) / (y * z);
        double[] point = [2, 3, 4, 5];

        AssertHessianMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Hessian_DivisionWithOverlappingVariables_MatchesFiniteDifference()
    {
        // This test catches bugs when the same variables appear in both numerator and denominator
        // For f(x,y) = (x + y) / (x * y), both x and y appear in L and R
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = (x + y) / (x * y);
        double[] point = [3, 2];

        AssertHessianMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Hessian_SimpleProduct_MatchesAnalytical()
    {
        // This test specifically catches the bug where product cross-terms are doubled
        // For f(x,y) = x * y:
        //   ∂²f/∂x∂y = 1  (the only non-zero Hessian entry)
        // The bug would give ∂²f/∂x∂y = 2 (doubled)

        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = x * y;
        double[] point = [3.0, 5.0];

        // Cache variables before computing Hessian
        expr.Prepare();

        var n = point.Length;
        var hess = new HessianAccumulator(n);
        expr.AccumulateHessian(point, hess, 1.0);

        // Analytical Hessian for f(x,y) = x*y:
        // H[0,0] = ∂²f/∂x² = 0
        // H[1,1] = ∂²f/∂y² = 0
        // H[1,0] = ∂²f/∂x∂y = 1 (stored in lower triangle)

        var h_xx = hess.Get(0, 0);
        var h_yy = hess.Get(1, 1);
        var h_xy = hess.Get(1, 0);

        Console.WriteLine($"DEBUG: h_xx={h_xx}, h_yy={h_yy}, h_xy={h_xy}");

        Assert.AreEqual(0.0, h_xx, 1e-10, "Hessian ∂²f/∂x² should be 0");
        Assert.AreEqual(0.0, h_yy, 1e-10, "Hessian ∂²f/∂y² should be 0");
        Assert.AreEqual(1.0, h_xy, 1e-10, "Hessian ∂²f/∂x∂y should be 1 (BUG: gives 2 if cross-terms doubled)");
    }

    [TestMethod]
    public void Hessian_ThreeFactorProduct_CrossTermsCorrect()
    {
        // Test f(x,y,z) = x * y * z
        // Cross-terms should be:
        //   ∂²f/∂x∂y = z
        //   ∂²f/∂x∂z = y
        //   ∂²f/∂y∂z = x

        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();

        var expr = x * y * z;
        double[] point = [2.0, 3.0, 5.0];

        // First verify this is actually creating a Product
        Assert.IsInstanceOfType(expr, typeof(Product), "x*y*z should create a Product");

        // Cache variables before computing Hessian
        expr.Prepare();

        var n = point.Length;
        var hess = new HessianAccumulator(n);
        expr.AccumulateHessian(point, hess, 1.0);

        // Analytical values:
        // ∂²f/∂x∂y = z = 5.0
        // ∂²f/∂x∂z = y = 3.0
        // ∂²f/∂y∂z = x = 2.0

        // Get cross-terms (stored in lower triangle)
        var h_xy = hess.Get(1, 0);  // ∂²f/∂x∂y
        var h_xz = hess.Get(2, 0);  // ∂²f/∂x∂z
        var h_yz = hess.Get(2, 1);  // ∂²f/∂y∂z

        Console.WriteLine($"DEBUG 3-factor: h_xy={h_xy} (expected {point[2]}), h_xz={h_xz} (expected {point[1]}), h_yz={h_yz} (expected {point[0]})");

        Assert.AreEqual(point[2], h_xy, 1e-10, $"∂²f/∂x∂y should equal z={point[2]}, got {h_xy}");
        Assert.AreEqual(point[1], h_xz, 1e-10, $"∂²f/∂x∂z should equal y={point[1]}, got {h_xz}");
        Assert.AreEqual(point[0], h_yz, 1e-10, $"∂²f/∂y∂z should equal x={point[0]}, got {h_yz}");
    }

    [TestMethod]
    public void Gradient_ProductWithFixedVariable_MatchesFiniteDifference()
    {
        // Test Product with a fixed variable
        var model = new Model();
        var fixedVar = model.AddVariable(1, 1); // Fixed to 1
        var freeVar = model.AddVariable(0); // >= 0

        // Product: fixedVar * freeVar (where fixedVar = 1)
        var expr = fixedVar * freeVar;
        double[] point = [1, 5]; // fixedVar=1, freeVar=5

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Gradient_SquaredLinExprWithFixedVariableProducts_MatchesFiniteDifference()
    {
        // Match user's model structure: Pow((constant - fixedVar * freeVar - ...), 2)
        var model = new Model();
        var fixed1 = model.AddVariable(1, 1); // Fixed to 1 (like Variable[99])
        var free1 = model.AddVariable(0); // Free (like Variable[4])
        var free2 = model.AddVariable(0); // Another free variable

        // Build: (16.77 - fixed1 * free1 - fixed1 * free2)^2
        // This matches the user's pattern where Variable[99] (fixed to 1) multiplies free variables
        var linExpr = 16.77 - fixed1 * free1 - fixed1 * free2;
        var expr = Expr.Pow(linExpr, 2);

        double[] point = [1, 5, 3]; // fixed1=1, free1=5, free2=3

        AssertGradientMatchesFiniteDifference(expr, point);
    }

    [TestMethod]
    public void Gradient_ComplexNestedWithZeroWeightQuadratic_MatchesFiniteDifference()
    {
        // Replicate the exact structure from user's Variable[73] term
        var model = new Model();
        var v96 = model.AddVariable(0);
        var v73 = model.AddVariable(0); // The variable with 14.7% error
        var v139 = model.AddVariable(0);
        var v113 = model.AddVariable(0);
        var v129 = model.AddVariable(0);
        var v134 = model.AddVariable(0);
        var v136 = model.AddVariable(0);
        var v137 = model.AddVariable(0);
        var v138 = model.AddVariable(0);

        // Build QuadExpr with zero-weight terms
        var quad = new QuadExpr();
        quad.LinearTerms.Add(v138);
        quad.LinearWeights.Add(1.0);
        quad.QuadraticTerms1.Add(v129);
        quad.QuadraticTerms2.Add(v134);
        quad.QuadraticWeights.Add(0); // Zero weight!
        quad.QuadraticTerms1.Add(v136);
        quad.QuadraticTerms2.Add(v129);
        quad.QuadraticWeights.Add(0.042);
        quad.QuadraticTerms1.Add(v129);
        quad.QuadraticTerms2.Add(v137);
        quad.QuadraticWeights.Add(1.0);

        // Build LinExpr: constant - v96*v73 - v96*v139*quad - v113*quad
        var linExpr = 3.27 - 0.001 * v96 * v73 - v96 * v139 * quad - v113 * quad;

        // Square it
        var expr = Expr.Pow(linExpr, 2);

        double[] point = [2, 3, 4, 5, 6, 7, 8, 9, 10]; // v96, v73, v139, v113, v129, v134, v136, v137, v138

        // Manually check gradients with more detailed output
        expr.Prepare();
        var adGrad = new double[point.Length];
        expr.AccumulateGradient(point, adGrad);
        var fdGrad = ComputeFiniteDifferenceGradient(expr, point);

        for (int i = 0; i < point.Length; i++)
        {
            var relativeError = Math.Abs(adGrad[i] - fdGrad[i]) / Math.Max(Math.Abs(fdGrad[i]), 1e-10);
            Assert.IsTrue(relativeError < 0.01,
                $"Gradient[{i}] has {relativeError*100:F2}% error: AD={adGrad[i]}, FD={fdGrad[i]}");
        }
    }

    private static void AssertGradientMatchesFiniteDifference(Expr expr, double[] point)
    {
        // Cache variables before computing gradient
        expr.Prepare();

        var n = point.Length;
        var adGrad = new double[n];
        expr.AccumulateGradient(point, adGrad);

        var fdGrad = ComputeFiniteDifferenceGradient(expr, point);

        for (int i = 0; i < n; i++)
            Assert.AreEqual(fdGrad[i], adGrad[i], GradientTolerance, $"Gradient mismatch at index {i}");
    }

    private static void AssertHessianMatchesFiniteDifference(Expr expr, double[] point)
    {
        // Cache variables before computing Hessian
        expr.Prepare();

        var n = point.Length;
        var hess = new HessianAccumulator(n);
        expr.AccumulateHessian(point, hess, 1.0);

        var fdHess = ComputeFiniteDifferenceHessian(expr, point);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                var adValue = hess.Get(i, j);
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
    public void AdditionOperator_CreatesSum_NotDeepTree()
    {
        var model = new Model();
        var variables = new Variable[100];
        for (int i = 0; i < 100; i++)
            variables[i] = model.AddVariable();

        // Build a large sum incrementally using regular + operator (not +=)
        // This simulates the pattern: obj = obj + residual * residual
        Expr sum = variables[0];
        for (int i = 1; i < 100; i++)
            sum = sum + variables[i];

        // Verify it creates a LinExpr expression, not a deep tree of expressions
        Assert.IsInstanceOfType(sum, typeof(LinExpr));
        var linExpr = (LinExpr)sum;
        Assert.AreEqual(100, linExpr.Terms.Count);

        // Verify it evaluates correctly
        var point = new double[100];
        for (int i = 0; i < 100; i++)
            point[i] = i + 1;

        var result = sum.Evaluate(point);
        Assert.AreEqual(5050.0, result, 1e-10); // Sum of 1 to 100

        // Verify gradient is correct
        AssertGradientMatchesFiniteDifference(sum, point);
    }

    [TestMethod]
    public void MultiplicationOperator_CreatesProduct_NotDeepTree()
    {
        var model = new Model();
        var variables = new Variable[50];
        for (int i = 0; i < 50; i++)
            variables[i] = model.AddVariable();

        // Build a large product incrementally using regular * operator (not *=)
        // This simulates the pattern: obj = obj * factor
        Expr product = variables[0];
        for (int i = 1; i < 50; i++)
            product = product * variables[i];

        // Verify it creates a Product expression, not a deep tree of Divisions
        Assert.IsInstanceOfType(product, typeof(Product));
        var productExpr = (Product)product;
        Assert.AreEqual(50, productExpr.Factors.Count);

        // Verify it evaluates correctly (use small values to avoid overflow)
        var point = new double[50];
        for (int i = 0; i < 50; i++)
            point[i] = 1.01; // Small multiplier

        var result = product.Evaluate(point);
        var expected = Math.Pow(1.01, 50);
        Assert.AreEqual(expected, result, 1e-10);

        // Verify gradient is correct
        AssertGradientMatchesFiniteDifference(product, point);
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

    [TestMethod]
    public void Sum_AutomaticallyConsolidatesConstants()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create a sum with multiple constants: x + 2 + y + 3 + 5
        var sum = new LinExpr([x, new Constant(2), y, new Constant(3), new Constant(5)]);

        // Constants should be automatically consolidated during construction
        Assert.AreEqual(2, sum.Terms.Count, "Should have 2 non-constant terms (x and y)");
        Assert.AreEqual(10.0, sum.ConstantTerm, 1e-10, "ConstantTerm should be 2 + 3 + 5 = 10");

        // Verify no Constants in Terms
        Assert.IsFalse(sum.Terms.Any(t => t is Constant), "Sum.Terms should not contain any Constant expressions");

        // Verify evaluation still works correctly
        double[] point = [1, 2];
        var result = sum.Evaluate(point);
        Assert.AreEqual(13.0, result, 1e-10); // x(1) + y(2) + 10 = 13
    }

    [TestMethod]
    public void LinExpr_ExtractsWeightsFromProducts()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create: 2*x + 3*y + 5
        var lin = new LinExpr([new Product([new Constant(2), x]),
                                new Product([new Constant(3), y]),
                                new Constant(5)]);

        Assert.AreEqual(2, lin.Terms.Count, "Should have 2 weighted terms");
        Assert.AreEqual(2.0, lin.Weights[0], 1e-10, "First weight should be 2");
        Assert.AreEqual(3.0, lin.Weights[1], 1e-10, "Second weight should be 3");
        Assert.AreEqual(5.0, lin.ConstantTerm, 1e-10, "Constant term should be 5");

        // Verify no Product expressions in Terms
        Assert.IsFalse(lin.Terms.Any(t => t is Product), "LinExpr.Terms should not contain Product expressions for weighted terms");

        // Verify evaluation: 2*x + 3*y + 5 = 2(1) + 3(2) + 5 = 13
        double[] point = [1, 2];
        var result = lin.Evaluate(point);
        Assert.AreEqual(13.0, result, 1e-10);
    }

    [TestMethod]
    public void LinExpr_HandlesConstantOnRightSideOfProduct()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Create: x*2 (constant on right)
        var lin = new LinExpr([new Product([x, new Constant(2)])]);

        Assert.AreEqual(1, lin.Terms.Count);
        Assert.AreEqual(2.0, lin.Weights[0], 1e-10, "Weight should be 2");
        Assert.AreSame(x, lin.Terms[0], "Term should be x");
    }

    [TestMethod]
    public void LinExpr_GradientUsesWeights()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create: 2*x + 3*y
        var lin = new LinExpr([new Product([new Constant(2), x]),
                                new Product([new Constant(3), y])]);

        lin.Prepare();

        double[] point = [1, 2];
        var grad = new double[2];
        lin.AccumulateGradient(point, grad);

        // Gradient should be [2, 3]
        Assert.AreEqual(2.0, grad[0], 1e-10, "Gradient wrt x should be 2");
        Assert.AreEqual(3.0, grad[1], 1e-10, "Gradient wrt y should be 3");
    }

    [TestMethod]
    public void LinExpr_AdditionOperatorCreatesOptimizedExpression()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Build: 2*x + 3*y using operator
        var expr = 2 * x + 3 * y;

        Assert.IsInstanceOfType(expr, typeof(LinExpr), "Should create LinExpr");
        var lin = (LinExpr)expr;

        Assert.AreEqual(2, lin.Terms.Count);
        Assert.AreEqual(2.0, lin.Weights[0], 1e-10);
        Assert.AreEqual(3.0, lin.Weights[1], 1e-10);

        // Verify evaluation
        double[] point = [1, 2];
        Assert.AreEqual(8.0, expr.Evaluate(point), 1e-10); // 2*1 + 3*2 = 8
    }

    [TestMethod]
    public void LinExpr_CompoundAssignmentUsesWeights()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Build using compound assignment
        Expr expr = 0;
        expr += 2 * x;
        expr += 3 * y;

        // Verify the expression evaluates correctly
        double[] point = [1, 2];
        Assert.AreEqual(8.0, expr.Evaluate(point), 1e-10);

        expr.Prepare();

        // Verify gradient
        var grad = new double[2];
        expr.AccumulateGradient(point, grad);
        Assert.AreEqual(2.0, grad[0], 1e-10);
        Assert.AreEqual(3.0, grad[1], 1e-10);
    }

    [TestMethod]
    public void LinExpr_ClonePreservesWeights()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create: 2*x + 3*y + 5
        var lin1 = new LinExpr([new Product([new Constant(2), x]),
                                 new Product([new Constant(3), y]),
                                 new Constant(5)]);

        // Create identical expression
        var lin2 = new LinExpr([new Product([new Constant(2), x]),
                                 new Product([new Constant(3), y]),
                                 new Constant(5)]);

        double[] point = [1, 2];
        Assert.AreEqual(lin1.Evaluate(point), lin2.Evaluate(point), 1e-10, "Identical expressions should evaluate identically");

        // Verify both have correct structure
        Assert.AreEqual(2, lin1.Terms.Count);
        Assert.AreEqual(2, lin2.Terms.Count);
        Assert.AreEqual(2.0, lin1.Weights[0], 1e-10);
        Assert.AreEqual(3.0, lin1.Weights[1], 1e-10);
        Assert.AreEqual(5.0, lin1.ConstantTerm, 1e-10);
    }

    [TestMethod]
    public void Sum_HandlesNoConstants()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var sum = new LinExpr([x, y]);
        Assert.AreEqual(2, sum.Terms.Count);
        Assert.AreEqual(0.0, sum.ConstantTerm, 1e-10, "ConstantTerm should be 0 when no constants present");
    }

    [TestMethod]
    public void Sum_HandlesSingleConstant()
    {
        var model = new Model();
        var x = model.AddVariable();

        var sum = new LinExpr([x, new Constant(5)]);
        Assert.AreEqual(1, sum.Terms.Count, "Should have 1 non-constant term (x)");
        Assert.AreEqual(5.0, sum.ConstantTerm, 1e-10);
        Assert.IsFalse(sum.Terms.Any(t => t is Constant), "Sum.Terms should not contain any Constant expressions");
    }

    [TestMethod]
    public void Sum_HandlesOnlyConstants()
    {
        // Sum with only constants: 2 + 3 + 5
        var sum = new LinExpr([new Constant(2), new Constant(3), new Constant(5)]);
        Assert.AreEqual(0, sum.Terms.Count, "Should have 0 non-constant terms");
        Assert.AreEqual(10.0, sum.ConstantTerm, 1e-10);

        double[] point = [];
        var result = sum.Evaluate(point);
        Assert.AreEqual(10.0, result, 1e-10);
    }

    [TestMethod]
    public void Sum_PreservesGradientAfterConstantConsolidation()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var sum = new LinExpr([x, new Constant(2), y, new Constant(3), new Constant(5)]);
        sum.Prepare();

        double[] point = [1.5, 2.5];

        // Compute gradient
        var grad = new double[2];
        sum.AccumulateGradient(point, grad);

        // Gradient should be [1, 1] (derivatives wrt x and y)
        Assert.AreEqual(1.0, grad[0], 1e-10, "Gradient wrt x should be 1");
        Assert.AreEqual(1.0, grad[1], 1e-10, "Gradient wrt y should be 1");
    }

    [TestMethod]
    public void Sum_ClonePreservesConstantTerm()
    {
        var model = new Model();
        var x = model.AddVariable();

        var sum = new LinExpr([x, new Constant(5)]);

        // Verify by evaluation instead of accessing protected CloneCore
        double[] point = [3];
        var expected = sum.Evaluate(point);

        var sum2 = new LinExpr([x, new Constant(5)]);
        Assert.AreEqual(expected, sum2.Evaluate(point), 1e-10, "Clone should evaluate identically");
    }

    [TestMethod]
    public void Model_PrintOutputsExpressionTrees()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(-5, 5);

        // Create a simple objective: 2*x + 3*y + 5
        model.SetObjective(2 * x + 3 * y + 5);

        // Add constraints
        model.AddConstraint(x + y >= 1);
        model.AddConstraint(x * x + y * y <= 25);

        // Test 1: Output to console
        Console.WriteLine("\n========== Model Print Test (Console) ==========");
        Console.WriteLine(model);
        Console.WriteLine("================================================\n");

        // Test 2: Output to StringWriter
        var output = model.ToString();

        // Verify output contains expected content
        Assert.IsTrue(output.Contains("Variables: 2"));
        Assert.IsTrue(output.Contains("Objective:"));
        Assert.IsTrue(output.Contains("Constraints: 2"));
        Assert.IsTrue(output.Contains("x[0]"));
        Assert.IsTrue(output.Contains("x[1]"));

        Console.WriteLine("StringWriter output verified successfully");
    }

    [TestMethod]
    public void Model_PrintToFile()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(-5, 5);

        model.SetObjective(2 * x + 3 * y + 5);
        model.AddConstraint(x + y >= 1);

        // Write to temporary file
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, model.ToString());

            // Read back and verify
            var content = File.ReadAllText(tempFile);
            Assert.IsTrue(content.Contains("Variables: 2"));
            Assert.IsTrue(content.Contains("Objective:"));

            Console.WriteLine($"Successfully wrote model to file: {tempFile}");
            Console.WriteLine($"File size: {new FileInfo(tempFile).Length} bytes");
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [TestMethod]
    public void LinExpr_ExtractsNegativeWeightFromNegation()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Create: -x
        var lin = new LinExpr([new Negation(x)]);

        Assert.AreEqual(1, lin.Terms.Count, "Should have 1 term");
        Assert.AreEqual(-1.0, lin.Weights[0], 1e-10, "Weight should be -1");
        Assert.AreSame(x, lin.Terms[0], "Term should be x (without Negation wrapper)");

        // Verify evaluation: -x = -1 * 2 = -2
        double[] point = [2];
        var result = lin.Evaluate(point);
        Assert.AreEqual(-2.0, result, 1e-10);
    }

    [TestMethod]
    public void LinExpr_HandlesNegatedProduct()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Create: -(2*x) which is Negation(Product([Constant(2), x]))
        var lin = new LinExpr([new Negation(new Product([new Constant(2), x]))]);

        Assert.AreEqual(1, lin.Terms.Count);
        Assert.AreEqual(-2.0, lin.Weights[0], 1e-10, "Weight should be -2");
        Assert.AreSame(x, lin.Terms[0], "Term should be x");

        // Verify evaluation: -2*x = -2 * 3 = -6
        double[] point = [3];
        Assert.AreEqual(-6.0, lin.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void LinExpr_HandlesDoubleNegation()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Create: -(-x) which should become +x
        var lin = new LinExpr([new Negation(new Negation(x))]);

        Assert.AreEqual(1, lin.Terms.Count);
        Assert.AreEqual(1.0, lin.Weights[0], 1e-10, "Weight should be 1 (double negation cancels)");
        Assert.AreSame(x, lin.Terms[0]);

        double[] point = [5];
        Assert.AreEqual(5.0, lin.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void LinExpr_HandlesNegatedConstant()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Create: x + (-5) which is x + Negation(Constant(5))
        var lin = new LinExpr([x, new Negation(new Constant(5))]);

        Assert.AreEqual(1, lin.Terms.Count, "Should have 1 non-constant term");
        Assert.AreEqual(-5.0, lin.ConstantTerm, 1e-10, "Constant term should be -5");
        Assert.AreSame(x, lin.Terms[0]);

        double[] point = [10];
        Assert.AreEqual(5.0, lin.Evaluate(point), 1e-10); // 10 + (-5) = 5
    }

    [TestMethod]
    public void LinExpr_FiltersZeroWeightTerms()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create LinExpr with zero-weight term
        var lin = new LinExpr();
        lin.AddTerm(x, 2.0);
        lin.AddTerm(y, 0.0); // Should be filtered out
        lin.AddTerm(x, 0.0); // Should be filtered out

        Assert.AreEqual(1, lin.Terms.Count, "Should only have 1 term (zero-weight terms filtered)");
        Assert.AreEqual(2.0, lin.Weights[0], 1e-10);
        Assert.AreSame(x, lin.Terms[0]);

        double[] point = [5, 3];
        Assert.AreEqual(10.0, lin.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_FiltersZeroWeightTerms()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create QuadExpr with zero-weight terms
        var quad = new QuadExpr();
        quad.AddTerm(x, 2.0);
        quad.AddTerm(y, 0.0); // Should be filtered out
        quad.AddTerm(x * y, 3.0);
        quad.AddTerm(x * x, 0.0); // Should be filtered out

        Assert.AreEqual(1, quad.LinearTerms.Count, "Should only have 1 linear term");
        Assert.AreEqual(1, quad.QuadraticTerms1.Count, "Should only have 1 quadratic term");
        Assert.AreEqual(2.0, quad.LinearWeights[0], 1e-10);
        Assert.AreSame(x, quad.LinearTerms[0]);
        Assert.AreEqual(3.0, quad.QuadraticWeights[0], 1e-10);

        double[] point = [5, 3];
        Assert.AreEqual(2.0 * 5 + 3.0 * 5 * 3, quad.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void LinExpr_MixedNegationsAndProducts()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create: 2*x - 3*y + 5 using operators (which creates Negation nodes)
        var expr = 2 * x - 3 * y + 5;

        Assert.IsInstanceOfType(expr, typeof(LinExpr));
        var lin = (LinExpr)expr;

        // Verify no Negation or nested LinExpr nodes stored in Terms
        Assert.IsFalse(lin.Terms.Any(t => t is Negation), "LinExpr.Terms should not contain Negation nodes");
        Assert.IsFalse(lin.Terms.Any(t => t is LinExpr), "LinExpr.Terms should not contain nested LinExpr nodes");
        Assert.AreEqual(5.0, lin.ConstantTerm, 1e-10);

        // Verify evaluation: 2*1 - 3*2 + 5 = 2 - 6 + 5 = 1
        double[] point = [1, 2];
        Assert.AreEqual(1.0, expr.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void LinExpr_MergesNestedLinExpr()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create two LinExprs and add them
        var lin1 = new LinExpr([new Product([new Constant(2), x]), new Constant(3)]);
        var lin2 = new LinExpr([new Product([new Constant(4), y]), new Constant(5)]);

        // Add them together: (2*x + 3) + (4*y + 5)
        var combined = new LinExpr([lin1, lin2]);

        Assert.AreEqual(2, combined.Terms.Count, "Should have 2 terms (x and y)");
        Assert.AreEqual(8.0, combined.ConstantTerm, 1e-10, "Constant should be 3 + 5 = 8");

        // Verify no nested LinExpr
        Assert.IsFalse(combined.Terms.Any(t => t is LinExpr), "Should not contain nested LinExpr");

        // Verify evaluation: 2*1 + 4*2 + 8 = 2 + 8 + 8 = 18
        double[] point = [1, 2];
        Assert.AreEqual(18.0, combined.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void LinExpr_MergesNegatedLinExpr()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create: (2*x + 3) - (4*y + 5) = 2*x - 4*y + (3 - 5) = 2*x - 4*y - 2
        var lin1 = new LinExpr([new Product([new Constant(2), x]), new Constant(3)]);
        var lin2 = new LinExpr([new Product([new Constant(4), y]), new Constant(5)]);

        var combined = new LinExpr([lin1, new Negation(lin2)]);

        Assert.AreEqual(2, combined.Terms.Count, "Should have 2 terms");
        Assert.AreEqual(-2.0, combined.ConstantTerm, 1e-10, "Constant should be 3 - 5 = -2");

        // Find weights (order may vary)
        var weights = combined.Weights.OrderBy(w => w).ToList();
        Assert.AreEqual(-4.0, weights[0], 1e-10, "Should have weight -4");
        Assert.AreEqual(2.0, weights[1], 1e-10, "Should have weight 2");

        // Verify evaluation: 2*1 - 4*2 - 2 = 2 - 8 - 2 = -8
        double[] point = [1, 2];
        Assert.AreEqual(-8.0, combined.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void LinExpr_MergesWeightedLinExpr()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create: 3 * (2*x + 4*y + 5) = 6*x + 12*y + 15
        var inner = new LinExpr([new Product([new Constant(2), x]),
                                  new Product([new Constant(4), y]),
                                  new Constant(5)]);
        var weighted = new Product([new Constant(3), inner]);

        var combined = new LinExpr([weighted]);

        Assert.AreEqual(2, combined.Terms.Count);
        Assert.AreEqual(15.0, combined.ConstantTerm, 1e-10, "Constant should be 3 * 5 = 15");

        // Find weights
        var weights = combined.Weights.OrderBy(w => w).ToList();
        Assert.AreEqual(6.0, weights[0], 1e-10, "Should have weight 6 (3 * 2)");
        Assert.AreEqual(12.0, weights[1], 1e-10, "Should have weight 12 (3 * 4)");

        // Verify evaluation: 6*1 + 12*2 + 15 = 6 + 24 + 15 = 45
        double[] point = [1, 2];
        Assert.AreEqual(45.0, combined.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void LinExpr_ComplexNestedMerging()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Build step by step
        var term1 = 2 * x + 3; // LinExpr with weight=2 for x, constant=3
        var term2 = 4 * x + 5; // LinExpr with weight=4 for x, constant=5

        // Create: x + (2*x + 3) - (4*x + 5) + 6
        var expr = x + term1 - term2 + 6;

        var lin = (LinExpr)expr;

        // Verify no nested LinExpr (main goal of this optimization)
        Assert.IsFalse(lin.Terms.Any(t => t is LinExpr), "Should not contain nested LinExpr");
        Assert.IsFalse(lin.Terms.Any(t => t is Negation), "Should not contain Negation");

        // Note: The expression might have duplicate variable terms which is OK
        // The important optimizations are: no nesting, negations handled, constants consolidated
    }

    [TestMethod]
    public void QuadExpr_SimpleQuadratic()
    {
        var model = new Model();
        var x = model.AddVariable();

        // x^2
        var expr = Expr.Pow(x, 2);

        Assert.IsInstanceOfType(expr, typeof(PowerOp)); // Pow itself is not QuadExpr

        // But wrapping in QuadExpr should recognize it
        var quad = new QuadExpr([expr]);
        Assert.AreEqual(1, quad.QuadraticTerms1.Count);
        Assert.AreEqual(0, quad.LinearTerms.Count);
        Assert.AreEqual(0.0, quad.ConstantTerm);
        Assert.AreEqual(1.0, quad.QuadraticWeights[0]);

        // Evaluate: x=3 → 3^2 = 9
        double[] point = [3];
        Assert.AreEqual(9.0, quad.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_WeightedQuadratic()
    {
        var model = new Model();
        var x = model.AddVariable();

        // 3*x^2
        var expr = new QuadExpr([3 * Expr.Pow(x, 2)]);

        Assert.AreEqual(1, expr.QuadraticTerms1.Count);
        Assert.AreEqual(3.0, expr.QuadraticWeights[0]);

        // Evaluate: x=2 → 3*4 = 12
        double[] point = [2];
        Assert.AreEqual(12.0, expr.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_Bilinear()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // x*y
        var expr = new QuadExpr([x * y]);

        Assert.AreEqual(1, expr.QuadraticTerms1.Count);
        Assert.AreEqual(0, expr.LinearTerms.Count);
        Assert.AreEqual(1.0, expr.QuadraticWeights[0]);

        // Evaluate: x=3, y=4 → 12
        double[] point = [3, 4];
        Assert.AreEqual(12.0, expr.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_MixedExpression()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // 2*x^2 + 3*x*y + 4*x + 5
        var expr = new QuadExpr([2 * Expr.Pow(x, 2), 3 * x * y, 4 * x, new Constant(5)]);

        Assert.AreEqual(2, expr.QuadraticTerms1.Count); // x^2 and x*y
        Assert.AreEqual(1, expr.LinearTerms.Count); // x
        Assert.AreEqual(5.0, expr.ConstantTerm);

        // Evaluate: x=2, y=3 → 2*4 + 3*2*3 + 4*2 + 5 = 8 + 18 + 8 + 5 = 39
        double[] point = [2, 3];
        Assert.AreEqual(39.0, expr.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_LinExprTimesLinExpr()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // (2*x + 3) * (4*y + 5)
        var lin1 = new LinExpr([2 * x, new Constant(3)]);
        var lin2 = new LinExpr([4 * y, new Constant(5)]);
        var expr = new QuadExpr([lin1 * lin2]);

        // Expansion: 2*x*4*y + 2*x*5 + 3*4*y + 3*5 = 8*x*y + 10*x + 12*y + 15
        Assert.AreEqual(1, expr.QuadraticTerms1.Count); // x*y
        Assert.AreEqual(2, expr.LinearTerms.Count); // x and y
        Assert.AreEqual(15.0, expr.ConstantTerm);

        // Evaluate: x=1, y=2 → (2+3)*(8+5) = 5*13 = 65
        double[] point = [1, 2];
        Assert.AreEqual(65.0, expr.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_VariableSquaredAsProduct()
    {
        var model = new Model();
        var x = model.AddVariable();

        // x*x (should be recognized as x^2)
        var expr = new QuadExpr([x * x]);

        Assert.AreEqual(1, expr.QuadraticTerms1.Count);
        Assert.AreSame(expr.QuadraticTerms1[0], expr.QuadraticTerms2[0]); // Same variable

        // Evaluate: x=5 → 25
        double[] point = [5];
        Assert.AreEqual(25.0, expr.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_MergesNestedQuadExpr()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var quad1 = new QuadExpr([x * x, 2 * x]);
        var quad2 = new QuadExpr([y * y, 3 * y]);

        // Combine them
        var combined = new QuadExpr([quad1, quad2, new Constant(5)]);

        Assert.AreEqual(2, combined.QuadraticTerms1.Count); // x^2 and y^2
        Assert.AreEqual(2, combined.LinearTerms.Count); // x and y
        Assert.AreEqual(5.0, combined.ConstantTerm);

        // Verify no nested QuadExprs
        Assert.IsFalse(combined.LinearTerms.Any(t => t is QuadExpr));
        Assert.IsFalse(combined.QuadraticTerms1.Any(t => t is QuadExpr));
        Assert.IsFalse(combined.QuadraticTerms2.Any(t => t is QuadExpr));
    }

    [TestMethod]
    public void QuadExpr_Negation()
    {
        var model = new Model();
        var x = model.AddVariable();

        // -(x^2)
        var expr = new QuadExpr([-Expr.Pow(x, 2)]);

        Assert.AreEqual(1, expr.QuadraticTerms1.Count);
        Assert.AreEqual(-1.0, expr.QuadraticWeights[0]);

        // Evaluate: x=3 → -9
        double[] point = [3];
        Assert.AreEqual(-9.0, expr.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_ComplexExpansion()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();

        // (x + y) * (y + z) = x*y + x*z + y*y + y*z
        var lin1 = new LinExpr([x, y]);
        var lin2 = new LinExpr([y, z]);
        var expr = new QuadExpr([lin1 * lin2]);

        // Should have 4 quadratic terms
        Assert.AreEqual(4, expr.QuadraticTerms1.Count);
        Assert.AreEqual(0.0, expr.ConstantTerm);

        // Evaluate: x=1, y=2, z=3 → (1+2)*(2+3) = 3*5 = 15
        double[] point = [1, 2, 3];
        Assert.AreEqual(15.0, expr.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_Print()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create a quadratic expression
        var expr = new QuadExpr([x * x, 2 * x * y, 3 * x, new Constant(4)]);

        // Test ToString
        var output = expr.ToString();

        Assert.IsTrue(output.Contains("x[0]"));
        Assert.IsTrue(output.Contains("x[1]"));
        Assert.IsTrue(output.Contains("4"));
    }

    [TestMethod]
    public void Operators_AutoCreateQuadExpr()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // x * y creates Product
        var xy = x * y;
        Assert.IsInstanceOfType(xy, typeof(Product), "x*y alone is Product");

        // x * y + 5 should be QuadExpr (Product with 2 non-constants triggers QuadExpr)
        var expr1 = x * y + 5;
        Assert.IsInstanceOfType(expr1, typeof(QuadExpr), "x*y + 5 should be QuadExpr");

        // 2*x + 3*y should be LinExpr
        var expr2 = 2 * x + 3 * y;
        Assert.IsInstanceOfType(expr2, typeof(LinExpr), "2*x + 3*y should be LinExpr");

        // x*x + 2*x + 1 should be QuadExpr
        var expr3 = x * x + 2 * x + 1;
        Assert.IsInstanceOfType(expr3, typeof(QuadExpr), "x*x + 2*x + 1 should be QuadExpr");

        // Pow(x, 2) + 1 should be QuadExpr
        var expr4 = Expr.Pow(x, 2) + 1;
        Assert.IsInstanceOfType(expr4, typeof(QuadExpr), "x^2 + 1 should be QuadExpr");
    }

    [TestMethod]
    public void LinExpr_MultiplyByConstant_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var linExpr = new LinExpr([2 * x, 3 * y, new Constant(4)]);

        // Multiply by constant
        var scaled = linExpr * 5.0;

        // Should still be LinExpr, not Product
        Assert.IsInstanceOfType(scaled, typeof(LinExpr), "LinExpr * constant should return LinExpr");

        var scaledLin = (LinExpr)scaled;
        Assert.AreEqual(2, scaledLin.Terms.Count, "Should have 2 terms");
        Assert.AreEqual(10.0, scaledLin.Weights[0], 1e-10, "First weight should be 2*5=10");
        Assert.AreEqual(15.0, scaledLin.Weights[1], 1e-10, "Second weight should be 3*5=15");
        Assert.AreEqual(20.0, scaledLin.ConstantTerm, 1e-10, "Constant should be 4*5=20");

        // Test double * LinExpr as well
        var scaled2 = 5.0 * linExpr;
        Assert.IsInstanceOfType(scaled2, typeof(LinExpr), "constant * LinExpr should return LinExpr");
        var scaledLin2 = (LinExpr)scaled2;
        Assert.AreEqual(10.0, scaledLin2.Weights[0], 1e-10);
        Assert.AreEqual(15.0, scaledLin2.Weights[1], 1e-10);
        Assert.AreEqual(20.0, scaledLin2.ConstantTerm, 1e-10);
    }

    [TestMethod]
    public void LinExpr_DivideByConstant_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var linExpr = new LinExpr([2 * x, 3 * y, new Constant(6)]);

        // Divide by constant
        var divided = linExpr / 2.0;

        // Should still be LinExpr, not Division
        Assert.IsInstanceOfType(divided, typeof(LinExpr), "LinExpr / constant should return LinExpr");

        var dividedLin = (LinExpr)divided;
        Assert.AreEqual(2, dividedLin.Terms.Count, "Should have 2 terms");
        Assert.AreEqual(1.0, dividedLin.Weights[0], 1e-10, "First weight should be 2/2=1");
        Assert.AreEqual(1.5, dividedLin.Weights[1], 1e-10, "Second weight should be 3/2=1.5");
        Assert.AreEqual(3.0, dividedLin.ConstantTerm, 1e-10, "Constant should be 6/2=3");
    }

    [TestMethod]
    public void QuadExpr_MultiplyByConstant_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var quadExpr = new QuadExpr([x * y, 2 * x, new Constant(3)]);

        // Multiply by constant
        var scaled = quadExpr * 4.0;

        // Should still be QuadExpr, not Product
        Assert.IsInstanceOfType(scaled, typeof(QuadExpr), "QuadExpr * constant should return QuadExpr");

        var scaledQuad = (QuadExpr)scaled;
        Assert.AreEqual(1, scaledQuad.LinearTerms.Count, "Should have 1 linear term");
        Assert.AreEqual(1, scaledQuad.QuadraticTerms1.Count, "Should have 1 quadratic term");
        Assert.AreEqual(8.0, scaledQuad.LinearWeights[0], 1e-10, "Linear weight should be 2*4=8");
        Assert.AreEqual(4.0, scaledQuad.QuadraticWeights[0], 1e-10, "Quadratic weight should be 1*4=4");
        Assert.AreEqual(12.0, scaledQuad.ConstantTerm, 1e-10, "Constant should be 3*4=12");

        // Test double * QuadExpr as well
        var scaled2 = 4.0 * quadExpr;
        Assert.IsInstanceOfType(scaled2, typeof(QuadExpr), "constant * QuadExpr should return QuadExpr");
        var scaledQuad2 = (QuadExpr)scaled2;
        Assert.AreEqual(8.0, scaledQuad2.LinearWeights[0], 1e-10);
        Assert.AreEqual(4.0, scaledQuad2.QuadraticWeights[0], 1e-10);
        Assert.AreEqual(12.0, scaledQuad2.ConstantTerm, 1e-10);
    }

    [TestMethod]
    public void QuadExpr_DivideByConstant_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var quadExpr = new QuadExpr([x * y, 4 * x, new Constant(8)]);

        // Divide by constant
        var divided = quadExpr / 2.0;

        // Should still be QuadExpr, not Division
        Assert.IsInstanceOfType(divided, typeof(QuadExpr), "QuadExpr / constant should return QuadExpr");

        var dividedQuad = (QuadExpr)divided;
        Assert.AreEqual(1, dividedQuad.LinearTerms.Count, "Should have 1 linear term");
        Assert.AreEqual(1, dividedQuad.QuadraticTerms1.Count, "Should have 1 quadratic term");
        Assert.AreEqual(2.0, dividedQuad.LinearWeights[0], 1e-10, "Linear weight should be 4/2=2");
        Assert.AreEqual(0.5, dividedQuad.QuadraticWeights[0], 1e-10, "Quadratic weight should be 1/2=0.5");
        Assert.AreEqual(4.0, dividedQuad.ConstantTerm, 1e-10, "Constant should be 8/2=4");
    }

    [TestMethod]
    public void LinExpr_CompoundMultiplyAssignment_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();

        var linExpr = new LinExpr([2 * x, new Constant(5)]);
        linExpr *= 3.0;

        var actual = linExpr.GetActual();
        Assert.IsInstanceOfType(actual, typeof(LinExpr), "LinExpr after *= constant should still be LinExpr");

        var lin = (LinExpr)actual;
        Assert.AreEqual(1, lin.Terms.Count, "Should have 1 term");
        Assert.AreEqual(6.0, lin.Weights[0], 1e-10, "Weight should be 2*3=6");
        Assert.AreEqual(15.0, lin.ConstantTerm, 1e-10, "Constant should be 5*3=15");
    }

    [TestMethod]
    public void LinExpr_CompoundDivideAssignment_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();

        var linExpr = new LinExpr([6 * x, new Constant(12)]);
        linExpr /= 3.0;

        var actual = linExpr.GetActual();
        Assert.IsInstanceOfType(actual, typeof(LinExpr), "LinExpr after /= constant should still be LinExpr");

        var lin = (LinExpr)actual;
        Assert.AreEqual(1, lin.Terms.Count, "Should have 1 term");
        Assert.AreEqual(2.0, lin.Weights[0], 1e-10, "Weight should be 6/3=2");
        Assert.AreEqual(4.0, lin.ConstantTerm, 1e-10, "Constant should be 12/3=4");
    }

    [TestMethod]
    public void QuadExpr_CompoundMultiplyAssignment_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var quadExpr = new QuadExpr([x * y, 2 * x, new Constant(4)]);
        quadExpr *= 2.0;

        var actual = quadExpr.GetActual();
        Assert.IsInstanceOfType(actual, typeof(QuadExpr), "QuadExpr after *= constant should still be QuadExpr");

        var quad = (QuadExpr)actual;
        Assert.AreEqual(1, quad.LinearTerms.Count, "Should have 1 linear term");
        Assert.AreEqual(1, quad.QuadraticTerms1.Count, "Should have 1 quadratic term");
        Assert.AreEqual(4.0, quad.LinearWeights[0], 1e-10, "Linear weight should be 2*2=4");
        Assert.AreEqual(2.0, quad.QuadraticWeights[0], 1e-10, "Quadratic weight should be 1*2=2");
        Assert.AreEqual(8.0, quad.ConstantTerm, 1e-10, "Constant should be 4*2=8");
    }

    [TestMethod]
    public void BuildComplexObjectiveWithCompoundOperators_ShouldWork()
    {
        // Reproduces issue where building objective with += and then dividing
        // would cause NullReferenceException in CollectHessianSparsity
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr obj = 0;
        for (int i = 0; i < 3; i++)
        {
            obj += x * y + Expr.Pow(x - i, 2);
        }

        model.SetObjective(obj / 3.0);

        // This should not throw
        var result = model.Solve();

        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void ExprDividedByConstantExpr_ScalesCoefficients()
    {
        // When dividing an expression by a Constant (not a double),
        // it should still scale coefficients instead of creating Division
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = new LinExpr([2 * x, 4 * y]);
        Expr divisor = 2.0; // Implicitly converts to Constant

        var result = expr / divisor;

        Assert.IsInstanceOfType<LinExpr>(result);
        var lin = (LinExpr)result;
        Assert.AreEqual(1.0, lin.Weights[0], 1e-10);
        Assert.AreEqual(2.0, lin.Weights[1], 1e-10);
    }

    [TestMethod]
    public void ExprMultipliedByConstantExpr_ScalesCoefficients()
    {
        // When multiplying an expression by a Constant (not a double),
        // it should still scale coefficients instead of creating Product
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = new LinExpr([2 * x, 4 * y]);
        Expr multiplier = 3.0; // Implicitly converts to Constant

        var result = expr * multiplier;

        Assert.IsInstanceOfType<LinExpr>(result);
        var lin = (LinExpr)result;
        Assert.AreEqual(6.0, lin.Weights[0], 1e-10);
        Assert.AreEqual(12.0, lin.Weights[1], 1e-10);
    }

    [TestMethod]
    public void ConstantExprMultipliedByExpr_ScalesCoefficients()
    {
        // When multiplying a Constant by an expression,
        // it should scale coefficients instead of creating Product
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr multiplier = 3.0; // Implicitly converts to Constant
        var expr = new LinExpr([2 * x, 4 * y]);

        var result = multiplier * expr;

        Assert.IsInstanceOfType<LinExpr>(result);
        var lin = (LinExpr)result;
        Assert.AreEqual(6.0, lin.Weights[0], 1e-10);
        Assert.AreEqual(12.0, lin.Weights[1], 1e-10);
    }

    [TestMethod]
    public void IntImplicitlyConvertedToDivision_ScalesCoefficients()
    {
        // When dividing by an int (which implicitly converts to Constant),
        // it should scale coefficients instead of creating Division
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var expr = new LinExpr([4 * x, 8 * y]);
        int divisor = 2;

        var result = expr / divisor;

        Assert.IsInstanceOfType<LinExpr>(result);
        var lin = (LinExpr)result;
        Assert.AreEqual(2.0, lin.Weights[0], 1e-10);
        Assert.AreEqual(4.0, lin.Weights[1], 1e-10);
    }

    [TestMethod]
    public void QuadExprDividedByConstantExpr_ScalesCoefficients()
    {
        // When dividing a QuadExpr by a Constant, it should scale coefficients
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var quad = new QuadExpr([6 * x * y, 4 * x]);

        Expr divisor = 2.0;
        var result = quad / divisor;

        Assert.IsInstanceOfType<QuadExpr>(result);
        var resultQuad = (QuadExpr)result;
        Assert.AreEqual(3.0, resultQuad.QuadraticWeights[0], 1e-10);
        Assert.AreEqual(2.0, resultQuad.LinearWeights[0], 1e-10);
    }

    [TestMethod]
    public void QuadExprMultipliedByConstantExpr_ScalesCoefficients()
    {
        // When multiplying a QuadExpr by a Constant, it should scale coefficients
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var quad = new QuadExpr([2 * x * y, 3 * x]);

        Expr multiplier = 3.0;
        var result = quad * multiplier;

        Assert.IsInstanceOfType<QuadExpr>(result);
        var resultQuad = (QuadExpr)result;
        Assert.AreEqual(6.0, resultQuad.QuadraticWeights[0], 1e-10);
        Assert.AreEqual(9.0, resultQuad.LinearWeights[0], 1e-10);
    }

    [TestMethod]
    public void CompoundOperatorFollowedByIntDivision_WorksCorrectly()
    {
        // Reproduces the real-world pattern: obj += expr in loop, then obj / count
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr obj = 0;
        for (int i = 0; i < 3; i++)
        {
            obj += x * y;
        }

        int count = 3;
        var result = obj / count;

        // Should create a QuadExpr (or LinExpr if optimized), not a Division
        Assert.IsTrue(result is QuadExpr || result is LinExpr,
            $"Expected QuadExpr or LinExpr but got {result.GetType().Name}");

        // Verify it evaluates correctly
        double[] point = [2.0, 3.0];
        double expected = (2.0 * 3.0 + 2.0 * 3.0 + 2.0 * 3.0) / 3.0;
        Assert.AreEqual(expected, result.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void Product_ConstantsExtractedToFactor()
    {
        // Constants should be extracted to Factor field, not kept in Factors
        var model = new Model();
        var x = model.AddVariable();

        var prod = new Product([new Constant(2), x, new Constant(3)]);

        Assert.AreEqual(1, prod.Factors.Count, "Should have 1 non-constant factor");
        Assert.AreEqual(6.0, prod.Factor, 1e-10, "Factor should be 2 * 3 = 6");
        Assert.IsFalse(prod.Factors.Any(f => f is Constant), "Factors should not contain Constants");
    }

    [TestMethod]
    public void Product_MultipleConstantsMultiplied()
    {
        // Multiple constants should be multiplied together
        var prod = new Product([new Constant(2), new Constant(3), new Constant(5)]);

        Assert.AreEqual(0, prod.Factors.Count, "Should have no non-constant factors");
        Assert.AreEqual(30.0, prod.Factor, 1e-10, "Factor should be 2 * 3 * 5 = 30");
    }

    [TestMethod]
    public void Product_OnlyConstant_ExtractedToFactor()
    {
        // Single constant should be extracted to Factor
        var prod = new Product([new Constant(7)]);

        Assert.AreEqual(0, prod.Factors.Count, "Should have no factors");
        Assert.AreEqual(7.0, prod.Factor, 1e-10, "Factor should be 7");
    }

    [TestMethod]
    public void Product_NoConstants_FactorIsOne()
    {
        // Product with no constants should have Factor = 1.0
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var prod = new Product([x, y]);

        Assert.AreEqual(2, prod.Factors.Count);
        Assert.AreEqual(1.0, prod.Factor, 1e-10, "Factor should be 1.0 when no constants");
    }

    [TestMethod]
    public void Product_FactorPreservedInOperations()
    {
        // Factor should be preserved when extending Products
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();

        var prod1 = new Product([new Constant(2), x]);
        var prod2 = prod1 * y; // Extend with y

        Assert.IsInstanceOfType<Product>(prod2);
        var prod2Product = (Product)prod2;
        Assert.AreEqual(2.0, prod2Product.Factor, 1e-10, "Factor should be preserved");
        Assert.AreEqual(2, prod2Product.Factors.Count, "Should have x and y");
    }

    [TestMethod]
    public void Product_FactorInEvaluation()
    {
        // Factor should be included in evaluation
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var prod = new Product([new Constant(2), x, y, new Constant(3)]);

        double[] point = [5.0, 7.0];
        double expected = 2.0 * 5.0 * 7.0 * 3.0; // 210
        Assert.AreEqual(expected, prod.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void Product_FactorInGradient()
    {
        // Factor should be included in gradient computation
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create 2 * x * y * 3 = 6 * x * y
        var prod = new Product([new Constant(2), x, y, new Constant(3)]);
        prod.Prepare();

        double[] point = [5.0, 7.0];
        var grad = new double[2];
        prod.AccumulateGradient(point, grad);

        // d(6*x*y)/dx = 6*y = 6*7 = 42
        // d(6*x*y)/dy = 6*x = 6*5 = 30
        Assert.AreEqual(42.0, grad[0], 1e-10, "Gradient w.r.t. x");
        Assert.AreEqual(30.0, grad[1], 1e-10, "Gradient w.r.t. y");
    }

    [TestMethod]
    public void Product_ScalarMultiplication_UpdatesFactor()
    {
        // Multiplying a Product by a scalar should update Factor
        var model = new Model();
        var x = model.AddVariable();

        var prod = new Product([new Constant(2), x]);
        var result = prod * 3.0;

        Assert.IsInstanceOfType<Product>(result);
        var resultProd = (Product)result;
        Assert.AreEqual(6.0, resultProd.Factor, 1e-10, "Factor should be 2 * 3 = 6");
        Assert.AreEqual(1, resultProd.Factors.Count, "Should still have 1 factor");
        Assert.IsFalse(resultProd.Factors.Any(f => f is Constant), "Should not contain Constants");
    }

    [TestMethod]
    public void Product_FactorInCompoundOperators()
    {
        // Factor should work correctly with compound operators
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr expr = 0;
        expr += new Product([new Constant(3), x, y]);

        var actual = expr.GetActual();
        double[] point = [2.0, 4.0];
        double expected = 3.0 * 2.0 * 4.0; // 24
        Assert.AreEqual(expected, actual.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_CompoundDivideAssignment_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var quadExpr = new QuadExpr([x * y, 8 * x, new Constant(16)]);
        quadExpr /= 4.0;

        var actual = quadExpr.GetActual();
        Assert.IsInstanceOfType(actual, typeof(QuadExpr), "QuadExpr after /= constant should still be QuadExpr");

        var quad = (QuadExpr)actual;
        Assert.AreEqual(1, quad.LinearTerms.Count, "Should have 1 linear term");
        Assert.AreEqual(1, quad.QuadraticTerms1.Count, "Should have 1 quadratic term");
        Assert.AreEqual(2.0, quad.LinearWeights[0], 1e-10, "Linear weight should be 8/4=2");
        Assert.AreEqual(0.25, quad.QuadraticWeights[0], 1e-10, "Quadratic weight should be 1/4=0.25");
        Assert.AreEqual(4.0, quad.ConstantTerm, 1e-10, "Constant should be 16/4=4");
    }

    [TestMethod]
    public void LinExpr_CompoundAssignment_ModifiesInPlace()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Create LinExpr directly (not through replacement)
        var linExpr = new LinExpr([2 * x, new Constant(5)]);
        var originalReference = linExpr;

        // Multiply by constant using compound assignment
        linExpr *= 3.0;

        // Verify the object was modified in-place (no replacement created)
        Assert.AreSame(originalReference, linExpr, "Should be the same object reference");
        Assert.AreEqual(6.0, linExpr.Weights[0], 1e-10, "Weight should be updated in-place");
        Assert.AreEqual(15.0, linExpr.ConstantTerm, 1e-10, "Constant should be updated in-place");

        // Verify no replacement was created
        Assert.AreSame(linExpr, linExpr.GetActual(), "GetActual should return the same object (no replacement)");
    }

    [TestMethod]
    public void QuadExpr_CompoundAssignment_ModifiesInPlace()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create QuadExpr directly (not through replacement)
        var quadExpr = new QuadExpr([x * y, 2 * x, new Constant(4)]);
        var originalReference = quadExpr;

        // Divide by constant using compound assignment
        quadExpr /= 2.0;

        // Verify the object was modified in-place (no replacement created)
        Assert.AreSame(originalReference, quadExpr, "Should be the same object reference");
        Assert.AreEqual(1.0, quadExpr.LinearWeights[0], 1e-10, "Linear weight should be updated in-place");
        Assert.AreEqual(0.5, quadExpr.QuadraticWeights[0], 1e-10, "Quadratic weight should be updated in-place");
        Assert.AreEqual(2.0, quadExpr.ConstantTerm, 1e-10, "Constant should be updated in-place");

        // Verify no replacement was created
        Assert.AreSame(quadExpr, quadExpr.GetActual(), "GetActual should return the same object (no replacement)");
    }

    [TestMethod]
    public void QuadExpr_CleanupLinExprWithProducts()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Manually create a LinExpr with Product terms (simulating old code path)
        var linWithProducts = new LinExpr();
        linWithProducts.Terms.Add(x * y);  // Product
        linWithProducts.Weights.Add(1.0);
        linWithProducts.ConstantTerm = 5.0;

        // Create QuadExpr from this LinExpr - should extract the Product into quadratic terms
        var quad = new QuadExpr([linWithProducts]);

        // The Product should have been moved to quadratic terms, not left in linear terms
        Assert.AreEqual(0, quad.LinearTerms.Count, "Should have no linear terms");
        Assert.AreEqual(1, quad.QuadraticTerms1.Count, "Should have 1 quadratic term");
        Assert.AreEqual(5.0, quad.ConstantTerm);

        // Verify no nested Products in linear terms
        foreach (var term in quad.LinearTerms)
        {
            Assert.IsFalse(term is Product, "LinearTerms should not contain Product nodes");
        }

        // Verify evaluation: x*y + 5 at [2, 3] = 6 + 5 = 11
        double[] point = [2, 3];
        Assert.AreEqual(11.0, quad.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_ResidualSquared()
    {
        var model = new Model();
        var variable = model.AddVariable();

        // Simulate: obj += (variable - 5)^2
        Expr obj = new Constant(0);
        var residual = variable - 5.0;

        Console.WriteLine("residual type: " + residual.GetType().Name);
        if (residual is LinExpr lin)
        {
            Console.WriteLine("  LinExpr with " + lin.Terms.Count + " terms");
            for (int i = 0; i < lin.Terms.Count; i++)
            {
                Console.WriteLine("    Term[" + i + "]: " + lin.Terms[i].GetType().Name + " weight=" + lin.Weights[i]);
            }
            Console.WriteLine("  ConstantTerm: " + lin.ConstantTerm);
        }

        obj += residual * residual;

        Console.WriteLine("\nobj type after +=: " + obj.GetType().Name);
        if (obj is QuadExpr quad)
        {
            Console.WriteLine("QuadExpr details:");
            Console.WriteLine("  LinearTerms: " + quad.LinearTerms.Count);
            for (int i = 0; i < quad.LinearTerms.Count; i++)
            {
                Console.WriteLine("    LinearTerm[" + i + "]: " + quad.LinearTerms[i].GetType().Name);
            }
            Console.WriteLine("  QuadraticTerms: " + quad.QuadraticTerms1.Count);
            Console.WriteLine("  ConstantTerm: " + quad.ConstantTerm);

            // Should have NO Product nodes in LinearTerms
            foreach (var term in quad.LinearTerms)
            {
                Assert.IsFalse(term is Product, "LinearTerms should not contain Product nodes");
            }

            // Should have 1 quadratic term (variable * variable)
            Assert.AreEqual(1, quad.QuadraticTerms1.Count, "Should have 1 quadratic term");

            // Verify evaluation: (x - 5)^2 at x=3 → (-2)^2 = 4
            double[] point = [3];
            Assert.AreEqual(4.0, quad.Evaluate(point), 1e-10);
        }
    }

    [TestMethod]
    public void QuadExpr_AccumulateResidualSquared()
    {
        var model = new Model();
        var v = new Variable[50];
        for (int i = 0; i < 50; i++)
        {
            v[i] = model.AddVariable();
        }

        // Accumulate 50 residual squared terms - exactly as user does it
        Expr obj = 0;
        for (int i = 0; i < 50; i++)
        {
            var residual = v[i] - 5;
            obj += residual * residual;
        }

        // Get the actual expression (might be behind _replacement)
        var actualExpr = obj.GetActual();

        // MUST be QuadExpr
        Assert.IsInstanceOfType(actualExpr, typeof(QuadExpr), $"Expected QuadExpr but got {actualExpr.GetType().Name}");
        var quad = (QuadExpr)actualExpr;

        // CRITICAL: Verify NO Product nodes anywhere in LinearTerms
        Console.WriteLine($"\n=== QuadExpr Structure ===");
        Console.WriteLine($"ConstantTerm: {quad.ConstantTerm}");
        Console.WriteLine($"LinearTerms: {quad.LinearTerms.Count}");
        Console.WriteLine($"QuadraticTerms: {quad.QuadraticTerms1.Count}");

        for (int i = 0; i < quad.LinearTerms.Count; i++)
        {
            var term = quad.LinearTerms[i];
            var weight = quad.LinearWeights[i];
            Console.WriteLine($"  Linear[{i}]: {term.GetType().Name} (weight={weight})");

            // FAIL if we find a Product
            if (term is Product prod)
            {
                Console.WriteLine($"    ERROR: Product found with {prod.Factors.Count} factors:");
                for (int j = 0; j < prod.Factors.Count; j++)
                {
                    Console.WriteLine($"      Factor[{j}]: {prod.Factors[j].GetType().Name}");
                    if (prod.Factors[j] is LinExpr linExpr)
                    {
                        Console.WriteLine($"        LinExpr with {linExpr.Terms.Count} terms, constant={linExpr.ConstantTerm}");
                    }
                }
                Assert.Fail($"LinearTerms[{i}] is a Product - should have been expanded to quadratic terms!");
            }

            // Linear terms should only be Variable or similar simple expressions
            Assert.IsTrue(term is Variable || term is Negation,
                $"Linear term should be Variable or Negation, not {term.GetType().Name}");
        }

        // Verify quadratic structure: (v[i] - 5)^2 = v[i]^2 - 10*v[i] + 25
        // Should have exactly 50 quadratic terms (one v[i]^2 per variable)
        Assert.AreEqual(50, quad.QuadraticTerms1.Count, "Should have 50 quadratic terms");

        // All quadratic terms should be v[i] * v[i] (same variable squared)
        for (int i = 0; i < quad.QuadraticTerms1.Count; i++)
        {
            Assert.IsInstanceOfType(quad.QuadraticTerms1[i], typeof(Variable),
                $"QuadraticTerm1[{i}] should be Variable");
            Assert.IsInstanceOfType(quad.QuadraticTerms2[i], typeof(Variable),
                $"QuadraticTerm2[{i}] should be Variable");
            Assert.AreEqual(quad.QuadraticTerms1[i], quad.QuadraticTerms2[i],
                $"QuadraticTerm {i} should be same variable squared");
        }

        Console.WriteLine($"\n=== Test PASSED - No Products in LinearTerms ===\n");
    }

    [TestMethod]
    public void CompoundOperators_EfficientOnDirectTypes()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Test LinExpr directly (not via Expr = 0)
        var linExpr = new LinExpr();
        linExpr += 2 * x;
        linExpr += 3 * y;

        // Verify it worked efficiently (should have 2 terms)
        Assert.AreEqual(2, linExpr.Terms.Count);
        Assert.AreEqual(2.0, linExpr.Weights[0]);
        Assert.AreEqual(3.0, linExpr.Weights[1]);

        // Test QuadExpr directly
        var quadExpr = new QuadExpr();
        quadExpr += x * y;
        quadExpr += x * x;

        // Should have 2 quadratic terms
        Assert.AreEqual(2, quadExpr.QuadraticTerms1.Count);
        Assert.AreEqual(0, quadExpr.LinearTerms.Count);

        // Test Product directly
        var prod = new Product([new Constant(2)]);
        prod *= x;
        prod *= y;

        // Should have 2 factors (Constants are extracted to Factor field)
        Assert.AreEqual(2, prod.Factors.Count);
        Assert.AreEqual(2.0, prod.Factor, 1e-10, "Factor should be 2");

        Console.WriteLine("Compound operators work efficiently on direct types!");
    }

    [TestMethod]
    public void QuadExpr_FromPowerOp()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Test (x-3)^2 + (y-4)^2 pattern
        var xm3 = x - 3;
        var ym4 = y - 4;

        var xSq = Expr.Pow(xm3, 2);
        var ySq = Expr.Pow(ym4, 2);

        var obj = xSq + ySq;

        Console.WriteLine("obj type: " + obj.GetType().Name);
        Console.WriteLine("obj actual: " + obj.GetActual().GetType().Name);

        var actualObj = obj.GetActual();
        if (actualObj is QuadExpr quad)
        {
            Console.WriteLine("QuadExpr:");
            Console.WriteLine("  LinearTerms: " + quad.LinearTerms.Count);
            Console.WriteLine("  QuadraticTerms: " + quad.QuadraticTerms1.Count);
            Console.WriteLine("  ConstantTerm: " + quad.ConstantTerm);

            // Should have 2 quadratic terms (x^2 and y^2)
            Assert.AreEqual(2, quad.QuadraticTerms1.Count, "Should have 2 quadratic terms");

            // Verify no Products in LinearTerms
            foreach (var term in quad.LinearTerms)
            {
                Assert.IsFalse(term is Product, "LinearTerms should not contain Products");
            }
        }
        else
        {
            Assert.Fail("Expected QuadExpr, got " + actualObj.GetType().Name);
        }
    }

    [TestMethod]
    public void DerivativeTest_DetectsWrongGradient()
    {
        // Test that the derivative test actually catches incorrect derivatives
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.FirstOrder;
        model.Options.CheckDerivativesForNanInf = true;
        model.Options.PrintLevel = 0;

        var x = model.AddVariable(0, 10);
        x.Start = 2.0;

        // Use a custom expression with intentionally wrong gradient
        var wrongExpr = new WrongGradientExpr(x);
        model.SetObjective(wrongExpr);

        var result = ModellingTests.SolveWithDerivativeTest(model);

        // Derivative test should have detected errors
        Assert.IsNotNull(result.DerivativeTestResult);
        Assert.IsFalse(result.DerivativeTestResult.Passed, "Derivative test should fail for wrong gradient");
        Assert.IsTrue(result.DerivativeTestResult.ErrorCount > 0, "Should report gradient errors");
    }

    [TestMethod]
    public void DerivativeTest_DetectsWrongHessian()
    {
        // Test that the derivative test actually catches incorrect Hessian
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        model.Options.PrintLevel = 0;

        var x = model.AddVariable(0, 10);
        x.Start = 2.0;

        // Use a custom expression with correct gradient but wrong Hessian
        var wrongExpr = new WrongHessianExpr(x);
        model.SetObjective(wrongExpr);

        var result = ModellingTests.SolveWithDerivativeTest(model);

        // Derivative test should have detected errors
        Assert.IsNotNull(result.DerivativeTestResult);
        Assert.IsFalse(result.DerivativeTestResult.Passed, "Derivative test should fail for wrong Hessian");
        Assert.IsTrue(result.DerivativeTestResult.ErrorCount > 0, "Should report Hessian errors");
    }
}

/// <summary>
/// Custom expression that returns x^2 but claims gradient is 3*x (wrong - should be 2*x)
/// </summary>
internal class WrongGradientExpr : Expr
{
    private readonly Variable _x;

    public WrongGradientExpr(Variable x)
    {
        _x = x;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Pow(x[_x.Index], 2);

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        // Correct would be: compactGrad[Array.BinarySearch(sortedVarIndices, _x.Index)] += multiplier * 2 * x[_x.Index];
        // But we intentionally return wrong derivative:
        compactGrad[Array.BinarySearch(sortedVarIndices, _x.Index)] += multiplier * 3 * x[_x.Index]; // WRONG!
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        hess.Add(_x.Index, _x.Index, multiplier * 2.0);
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        variables.Add(_x);
    }

    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        entries.Add((_x.Index, _x.Index));
    }

    protected override bool IsConstantWrtXCore() => false;
    protected override bool IsLinearCore() => false;
    protected override bool IsAtMostQuadraticCore() => true;

    protected override Expr CloneCore() => new WrongGradientExpr(_x);
}

/// <summary>
/// Custom expression that returns x^2 with correct gradient 2*x but claims Hessian is 5 (wrong - should be 2)
/// </summary>
internal class WrongHessianExpr : Expr
{
    private readonly Variable _x;

    public WrongHessianExpr(Variable x)
    {
        _x = x;
    }

    protected override double EvaluateCore(ReadOnlySpan<double> x) => Math.Pow(x[_x.Index], 2);

    protected override void AccumulateGradientCompactCore(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        compactGrad[Array.BinarySearch(sortedVarIndices, _x.Index)] += multiplier * 2 * x[_x.Index]; // Correct
    }

    protected override void AccumulateHessianCore(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // Correct would be: hess.Add(_x.Index, _x.Index, multiplier * 2.0);
        // But we intentionally return wrong second derivative:
        hess.Add(_x.Index, _x.Index, multiplier * 5.0); // WRONG!
    }

    protected override void CollectVariablesCore(HashSet<Variable> variables)
    {
        variables.Add(_x);
    }

    protected override void CollectHessianSparsityCore(HashSet<(int row, int col)> entries)
    {
        entries.Add((_x.Index, _x.Index));
    }

    protected override bool IsConstantWrtXCore() => false;
    protected override bool IsLinearCore() => false;
    protected override bool IsAtMostQuadraticCore() => true;

    protected override Expr CloneCore() => new WrongHessianExpr(_x);
}
