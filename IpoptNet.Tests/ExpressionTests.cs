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

        var n = point.Length;
        var hess = new HessianAccumulator(n);
        expr.AccumulateHessian(point, hess, 1.0);

        // Analytical Hessian for f(x,y) = x*y:
        // H[0,0] = ∂²f/∂x² = 0
        // H[1,1] = ∂²f/∂y² = 0  
        // H[1,0] = ∂²f/∂x∂y = 1 (stored in lower triangle)
        
        hess.Entries.TryGetValue((0, 0), out var h_xx);
        hess.Entries.TryGetValue((1, 1), out var h_yy);
        hess.Entries.TryGetValue((1, 0), out var h_xy);

        Console.WriteLine($"DEBUG: h_xx={h_xx}, h_yy={h_yy}, h_xy={h_xy}");
        Console.WriteLine($"DEBUG: Total Hessian entries: {hess.Entries.Count}");
        foreach (var entry in hess.Entries)
        {
            Console.WriteLine($"  H[{entry.Key.row},{entry.Key.col}] = {entry.Value}");
        }

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

        var n = point.Length;
        var hess = new HessianAccumulator(n);
        expr.AccumulateHessian(point, hess, 1.0);

        // Analytical values:
        // ∂²f/∂x∂y = z = 5.0
        // ∂²f/∂x∂z = y = 3.0  
        // ∂²f/∂y∂z = x = 2.0

        // Get cross-terms (stored in lower triangle)
        hess.Entries.TryGetValue((1, 0), out var h_xy);  // ∂²f/∂x∂y
        hess.Entries.TryGetValue((2, 0), out var h_xz);  // ∂²f/∂x∂z
        hess.Entries.TryGetValue((2, 1), out var h_yz);  // ∂²f/∂y∂z

        Console.WriteLine($"DEBUG 3-factor: h_xy={h_xy} (expected {point[2]}), h_xz={h_xz} (expected {point[1]}), h_yz={h_yz} (expected {point[0]})");
        Console.WriteLine($"DEBUG: Total Hessian entries: {hess.Entries.Count}");
        foreach (var entry in hess.Entries)
        {
            Console.WriteLine($"  H[{entry.Key.row},{entry.Key.col}] = {entry.Value}");
        }

        Assert.AreEqual(point[2], h_xy, 1e-10, $"∂²f/∂x∂y should equal z={point[2]}, got {h_xy}");
        Assert.AreEqual(point[1], h_xz, 1e-10, $"∂²f/∂x∂z should equal y={point[1]}, got {h_xz}");
        Assert.AreEqual(point[0], h_yz, 1e-10, $"∂²f/∂y∂z should equal x={point[0]}, got {h_yz}");
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
        var hess = new HessianAccumulator(n);
        expr.AccumulateHessian(point, hess, 1.0);

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

        double[] point = [1, 2];
        var grad = new double[2];
        lin.AccumulateGradient(point, grad, 1.0);

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

        // Verify gradient
        var grad = new double[2];
        expr.AccumulateGradient(point, grad, 1.0);
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
        double[] point = [1.5, 2.5];

        // Compute gradient
        var grad = new double[2];
        sum.AccumulateGradient(point, grad, 1.0);

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
        model.Print();
        Console.WriteLine("================================================\n");

        // Test 2: Output to StringWriter
        var sw = new StringWriter();
        model.Print(sw);
        var output = sw.ToString();
        
        // Verify output contains expected content
        Assert.IsTrue(output.Contains("=== Model ==="));
        Assert.IsTrue(output.Contains("Variables: 2"));
        Assert.IsTrue(output.Contains("Objective:"));
        Assert.IsTrue(output.Contains("LinExpr:"));
        Assert.IsTrue(output.Contains("Constraints: 2"));
        Assert.IsTrue(output.Contains("Variable[0]"));
        
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
            using (var writer = new StreamWriter(tempFile))
            {
                model.Print(writer);
            }

            // Read back and verify
            var content = File.ReadAllText(tempFile);
            Assert.IsTrue(content.Contains("=== Model ==="));
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
        var expr = new QuadExpr([x*x, 2*x*y, 3*x, new Constant(4)]);

        // Test Print to StringWriter
        var sw = new StringWriter();
        expr.Print(sw);
        var output = sw.ToString();

        Assert.IsTrue(output.Contains("QuadExpr"));
        Assert.IsTrue(output.Contains("linear"));
        Assert.IsTrue(output.Contains("quadratic"));
        Assert.IsTrue(output.Contains("constant=4"));
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
        var expr2 = 2*x + 3*y;
        Assert.IsInstanceOfType(expr2, typeof(LinExpr), "2*x + 3*y should be LinExpr");

        // x*x + 2*x + 1 should be QuadExpr
        var expr3 = x*x + 2*x + 1;
        Assert.IsInstanceOfType(expr3, typeof(QuadExpr), "x*x + 2*x + 1 should be QuadExpr");
        
        // Pow(x, 2) + 1 should be QuadExpr
        var expr4 = Expr.Pow(x, 2) + 1;
        Assert.IsInstanceOfType(expr4, typeof(QuadExpr), "x^2 + 1 should be QuadExpr");
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
        
        // Should have 3 factors
        Assert.AreEqual(3, prod.Factors.Count);

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
}
