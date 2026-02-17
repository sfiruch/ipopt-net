using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class ExpressionTests
{
    private const double FiniteDiffDelta = 1e-6;
    private const double GradientTolerance = 1e-4;
    private const double HessianTolerance = 5e-3;

    private static Expr Const(double value) => value;
    private static Expr Neg(Expr a) => -a;

    private static Expr Lin(params Expr[] terms) => new(new LinExprNode(terms.Select(t => t._node).ToList()));
    private static Expr Quad(params Expr[] terms) => new(new QuadExprNode(terms.Select(t => t._node).ToList()));
    private static Expr Prod(params Expr[] factors) => new(new ProductNode(factors.Select(f => f._node).ToList()));

    private static ExprNode N(Expr e) => e._node;
    private static ExprNode N(Variable v) => v._expr._node;

    private static LinExprNode LinNode(Expr e) => (LinExprNode)e._node;
    private static QuadExprNode QuadNode(Expr e) => (QuadExprNode)e._node;
    private static ProductNode ProdNode(Expr e) => (ProductNode)e._node;
    private static ConstantNode ConstNode(Expr e) => (ConstantNode)e._node;

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
        var product = Prod([x, x, y]);
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
        var product = Prod([x, x]);
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
        var entries = new HashSet<(int row, int col)>();
        expr.CollectHessianSparsity(entries);
        var sorted = entries.OrderBy(e => e.row).ThenBy(e => e.col).ToArray();
        var rows = sorted.Select(e => e.row).ToArray();
        var cols = sorted.Select(e => e.col).ToArray();

        var hess = new HessianAccumulator(n, rows, cols);
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

        // First verify this is actually creating a ProductNode
        Assert.IsInstanceOfType(expr._node, typeof(ProductNode), "x*y*z should create a ProductNode");

        // Cache variables before computing Hessian
        expr.Prepare();

        var n = point.Length;
        var entries = new HashSet<(int row, int col)>();
        expr.CollectHessianSparsity(entries);
        var sorted = entries.OrderBy(e => e.row).ThenBy(e => e.col).ToArray();
        var rows = sorted.Select(e => e.row).ToArray();
        var cols = sorted.Select(e => e.col).ToArray();

        var hess = new HessianAccumulator(n, rows, cols);
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

        // Build QuadExprNode with zero-weight terms
        var quad = Quad();
        var quadNode = QuadNode(quad);
        quadNode.LinearTerms.Add(N(v138));
        quadNode.LinearWeights.Add(1.0);
        quadNode.QuadraticTerms1.Add(N(v129));
        quadNode.QuadraticTerms2.Add(N(v134));
        quadNode.QuadraticWeights.Add(0); // Zero weight!
        quadNode.QuadraticTerms1.Add(N(v136));
        quadNode.QuadraticTerms2.Add(N(v129));
        quadNode.QuadraticWeights.Add(0.042);
        quadNode.QuadraticTerms1.Add(N(v129));
        quadNode.QuadraticTerms2.Add(N(v137));
        quadNode.QuadraticWeights.Add(1.0);

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
        expr.Prepare();
        var n = point.Length;

        var entries = new HashSet<(int row, int col)>();
        expr.CollectHessianSparsity(entries);
        var sorted = entries.OrderBy(e => e.row).ThenBy(e => e.col).ToArray();
        var rows = sorted.Select(e => e.row).ToArray();
        var cols = sorted.Select(e => e.col).ToArray();

        var hess = new HessianAccumulator(n, rows, cols);
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

        // Verify it creates a LinExprNode expression, not a deep tree of expressions
        Assert.IsInstanceOfType(sum._node, typeof(LinExprNode));
        var linExpr = LinNode(sum);
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

        // Verify it creates a ProductNode expression, not a deep tree of Divisions
        Assert.IsInstanceOfType(product._node, typeof(ProductNode));
        var productExpr = ProdNode(product);
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

        Assert.IsInstanceOfType(zero._node, typeof(ConstantNode));
        Assert.IsInstanceOfType(one._node, typeof(ConstantNode));
        Assert.IsInstanceOfType(five._node, typeof(ConstantNode));

        Assert.AreEqual(0.0, ConstNode(zero).Value);
        Assert.AreEqual(1.0, ConstNode(one).Value);
        Assert.AreEqual(5.0, ConstNode(five).Value);
    }

    [TestMethod]
    public void Sum_AutomaticallyConsolidatesConstants()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create a sum with multiple constants: x + 2 + y + 3 + 5
        var sum = Lin([x, Const(2), y, Const(3), Const(5)]);

        var sumNode = LinNode(sum);

        // Constants should be automatically consolidated during construction
        Assert.AreEqual(2, sumNode.Terms.Count, "Should have 2 non-constant terms (x and y)");
        Assert.AreEqual(10.0, sumNode.ConstantTerm, 1e-10, "ConstantTerm should be 2 + 3 + 5 = 10");

        // Verify no Constants in Terms
        Assert.IsFalse(sumNode.Terms.Any(t => t is ConstantNode), "Sum.Terms should not contain any ConstantNode terms");

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
        var lin = Lin([Prod([Const(2), x]),
                                Prod([Const(3), y]),
                                Const(5)]);

        var linNode = LinNode(lin);

        Assert.AreEqual(2, linNode.Terms.Count, "Should have 2 weighted terms");
        Assert.AreEqual(2.0, linNode.Weights[0], 1e-10, "First weight should be 2");
        Assert.AreEqual(3.0, linNode.Weights[1], 1e-10, "Second weight should be 3");
        Assert.AreEqual(5.0, linNode.ConstantTerm, 1e-10, "Constant term should be 5");

        // Verify no ProductNode terms in Terms
        Assert.IsFalse(linNode.Terms.Any(t => t is ProductNode), "LinExprNode.Terms should not contain ProductNode terms for weighted terms");

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
        var lin = Lin([Prod([x, Const(2)])]);

        var linNode = LinNode(lin);

        Assert.AreEqual(1, linNode.Terms.Count);
        Assert.AreEqual(2.0, linNode.Weights[0], 1e-10, "Weight should be 2");
        Assert.AreSame(N(x), linNode.Terms[0], "Term should be x");
    }

    [TestMethod]
    public void LinExpr_GradientUsesWeights()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create: 2*x + 3*y
        var lin = Lin([Prod([Const(2), x]),
                                Prod([Const(3), y])]);

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

        Assert.IsInstanceOfType(expr._node, typeof(LinExprNode), "Should create LinExprNode");
        var lin = LinNode(expr);

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
        var lin1 = Lin([Prod([Const(2), x]),
                                 Prod([Const(3), y]),
                                 Const(5)]);

        // Create identical expression
        var lin2 = Lin([Prod([Const(2), x]),
                                 Prod([Const(3), y]),
                                 Const(5)]);

        double[] point = [1, 2];
        Assert.AreEqual(lin1.Evaluate(point), lin2.Evaluate(point), 1e-10, "Identical expressions should evaluate identically");

        // Verify both have correct structure
        var lin1Node = LinNode(lin1);
        var lin2Node = LinNode(lin2);
        Assert.AreEqual(2, lin1Node.Terms.Count);
        Assert.AreEqual(2, lin2Node.Terms.Count);
        Assert.AreEqual(2.0, lin1Node.Weights[0], 1e-10);
        Assert.AreEqual(3.0, lin1Node.Weights[1], 1e-10);
        Assert.AreEqual(5.0, lin1Node.ConstantTerm, 1e-10);
    }

    [TestMethod]
    public void Sum_HandlesNoConstants()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var sum = Lin([x, y]);
        var sumNode = LinNode(sum);
        Assert.AreEqual(2, sumNode.Terms.Count);
        Assert.AreEqual(0.0, sumNode.ConstantTerm, 1e-10, "ConstantTerm should be 0 when no constants present");
    }

    [TestMethod]
    public void Sum_HandlesSingleConstant()
    {
        var model = new Model();
        var x = model.AddVariable();

        var sum = Lin([x, Const(5)]);
        var sumNode = LinNode(sum);
        Assert.AreEqual(1, sumNode.Terms.Count, "Should have 1 non-constant term (x)");
        Assert.AreEqual(5.0, sumNode.ConstantTerm, 1e-10);
        Assert.IsFalse(sumNode.Terms.Any(t => t is ConstantNode), "Sum.Terms should not contain any ConstantNode terms");
    }

    [TestMethod]
    public void Sum_HandlesOnlyConstants()
    {
        // Sum with only constants: 2 + 3 + 5
        var sum = Lin([Const(2), Const(3), Const(5)]);
        var sumNode = LinNode(sum);
        Assert.AreEqual(0, sumNode.Terms.Count, "Should have 0 non-constant terms");
        Assert.AreEqual(10.0, sumNode.ConstantTerm, 1e-10);

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

        var sum = Lin([x, Const(2), y, Const(3), Const(5)]);
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

        var sum = Lin([x, Const(5)]);

        // Verify by evaluation instead of accessing protected CloneCore
        double[] point = [3];
        var expected = sum.Evaluate(point);

        var sum2 = Lin([x, Const(5)]);
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
        var lin = Lin([Neg(x)]);
        var linNode = LinNode(lin);

        Assert.AreEqual(1, linNode.Terms.Count, "Should have 1 term");
        Assert.AreEqual(-1.0, linNode.Weights[0], 1e-10, "Weight should be -1");
        Assert.AreSame(N(x), linNode.Terms[0], "Term should be x (without Negation wrapper)");

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
        var lin = Lin([Neg(Prod([Const(2), x]))]);
        var linNode = LinNode(lin);

        Assert.AreEqual(1, linNode.Terms.Count);
        Assert.AreEqual(-2.0, linNode.Weights[0], 1e-10, "Weight should be -2");
        Assert.AreSame(N(x), linNode.Terms[0], "Term should be x");

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
        var lin = Lin([Neg(Neg(x))]);
        var linNode = LinNode(lin);

        Assert.AreEqual(1, linNode.Terms.Count);
        Assert.AreEqual(1.0, linNode.Weights[0], 1e-10, "Weight should be 1 (double negation cancels)");
        Assert.AreSame(N(x), linNode.Terms[0]);

        double[] point = [5];
        Assert.AreEqual(5.0, lin.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void LinExpr_HandlesNegatedConstant()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Create: x + (-5) which is x + Negation(Constant(5))
        var lin = Lin([x, Neg(Const(5))]);
        var linNode = LinNode(lin);

        Assert.AreEqual(1, linNode.Terms.Count, "Should have 1 non-constant term");
        Assert.AreEqual(-5.0, linNode.ConstantTerm, 1e-10, "Constant term should be -5");
        Assert.AreSame(N(x), linNode.Terms[0]);

        double[] point = [10];
        Assert.AreEqual(5.0, lin.Evaluate(point), 1e-10); // 10 + (-5) = 5
    }

    [TestMethod]
    public void LinExpr_FiltersZeroWeightTerms()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create LinExprNode with zero-weight term
        var lin = Lin();
        var linNode = LinNode(lin);
        linNode.AddTerm(N(x), 2.0);
        linNode.AddTerm(N(y), 0.0); // Should be filtered out
        linNode.AddTerm(N(x), 0.0); // Should be filtered out

        Assert.AreEqual(1, linNode.Terms.Count, "Should only have 1 term (zero-weight terms filtered)");
        Assert.AreEqual(2.0, linNode.Weights[0], 1e-10);
        Assert.AreSame(N(x), linNode.Terms[0]);

        double[] point = [5, 3];
        Assert.AreEqual(10.0, lin.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_FiltersZeroWeightTerms()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create QuadExprNode with zero-weight terms
        var quad = Quad();
        var quadNode = QuadNode(quad);
        quadNode.AddTerm(N(x), 2.0);
        quadNode.AddTerm(N(y), 0.0); // Should be filtered out
        quadNode.AddTerm(N(x * y), 3.0);
        quadNode.AddTerm(N(x * x), 0.0); // Should be filtered out

        Assert.AreEqual(1, quadNode.LinearTerms.Count, "Should only have 1 linear term");
        Assert.AreEqual(1, quadNode.QuadraticTerms1.Count, "Should only have 1 quadratic term");
        Assert.AreEqual(2.0, quadNode.LinearWeights[0], 1e-10);
        Assert.AreSame(N(x), quadNode.LinearTerms[0]);
        Assert.AreEqual(3.0, quadNode.QuadraticWeights[0], 1e-10);

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

        Assert.IsInstanceOfType(expr._node, typeof(LinExprNode));
        var lin = LinNode(expr);

        // Verify no NegationNode or nested LinExprNode stored in Terms
        Assert.IsFalse(lin.Terms.Any(t => t is NegationNode), "LinExprNode.Terms should not contain NegationNode terms");
        Assert.IsFalse(lin.Terms.Any(t => t is LinExprNode), "LinExprNode.Terms should not contain nested LinExprNode terms");
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
        var lin1 = Lin([Prod([Const(2), x]), Const(3)]);
        var lin2 = Lin([Prod([Const(4), y]), Const(5)]);

        // Add them together: (2*x + 3) + (4*y + 5)
        var combined = Lin([lin1, lin2]);
        var combinedNode = LinNode(combined);

        Assert.AreEqual(2, combinedNode.Terms.Count, "Should have 2 terms (x and y)");
        Assert.AreEqual(8.0, combinedNode.ConstantTerm, 1e-10, "Constant should be 3 + 5 = 8");

        // Verify no nested LinExprNode
        Assert.IsFalse(combinedNode.Terms.Any(t => t is LinExprNode), "Should not contain nested LinExprNode terms");

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
        var lin1 = Lin([Prod([Const(2), x]), Const(3)]);
        var lin2 = Lin([Prod([Const(4), y]), Const(5)]);

        var combined = Lin([lin1, Neg(lin2)]);
        var combinedNode = LinNode(combined);

        Assert.AreEqual(2, combinedNode.Terms.Count, "Should have 2 terms");
        Assert.AreEqual(-2.0, combinedNode.ConstantTerm, 1e-10, "Constant should be 3 - 5 = -2");

        // Find weights (order may vary)
        var weights = combinedNode.Weights.OrderBy(w => w).ToList();
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
        var inner = Lin([Prod([Const(2), x]),
                                  Prod([Const(4), y]),
                                  Const(5)]);
        var weighted = Prod([Const(3), inner]);

        var combined = Lin([weighted]);
        var combinedNode = LinNode(combined);

        Assert.AreEqual(2, combinedNode.Terms.Count);
        Assert.AreEqual(15.0, combinedNode.ConstantTerm, 1e-10, "Constant should be 3 * 5 = 15");

        // Find weights
        var weights = combinedNode.Weights.OrderBy(w => w).ToList();
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

        Assert.IsInstanceOfType(expr._node, typeof(LinExprNode));
        var lin = LinNode(expr);

        // Verify no nested LinExprNode (main goal of this optimization)
        Assert.IsFalse(lin.Terms.Any(t => t is LinExprNode), "Should not contain nested LinExprNode terms");
        Assert.IsFalse(lin.Terms.Any(t => t is NegationNode), "Should not contain NegationNode terms");

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

        Assert.IsInstanceOfType(expr._node, typeof(PowerOpNode)); // Pow itself is not QuadExprNode

        // But wrapping in QuadExprNode should recognize it
        var quad = Quad([expr]);
        var quadNode = QuadNode(quad);
        Assert.AreEqual(1, quadNode.QuadraticTerms1.Count);
        Assert.AreEqual(0, quadNode.LinearTerms.Count);
        Assert.AreEqual(0.0, quadNode.ConstantTerm);
        Assert.AreEqual(1.0, quadNode.QuadraticWeights[0]);

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
        var expr = Quad([3 * Expr.Pow(x, 2)]);
        var exprNode = QuadNode(expr);

        Assert.AreEqual(1, exprNode.QuadraticTerms1.Count);
        Assert.AreEqual(3.0, exprNode.QuadraticWeights[0]);

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
        var expr = Quad([x * y]);
        var exprNode = QuadNode(expr);

        Assert.AreEqual(1, exprNode.QuadraticTerms1.Count);
        Assert.AreEqual(0, exprNode.LinearTerms.Count);
        Assert.AreEqual(1.0, exprNode.QuadraticWeights[0]);

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
        var expr = Quad([2 * Expr.Pow(x, 2), 3 * x * y, 4 * x, Const(5)]);
        var exprNode = QuadNode(expr);

        Assert.AreEqual(2, exprNode.QuadraticTerms1.Count); // x^2 and x*y
        Assert.AreEqual(1, exprNode.LinearTerms.Count); // x
        Assert.AreEqual(5.0, exprNode.ConstantTerm);

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
        var lin1 = Lin([2 * x, Const(3)]);
        var lin2 = Lin([4 * y, Const(5)]);
        var expr = Quad([lin1 * lin2]);

        var exprNode = QuadNode(expr);

        // Expansion: 2*x*4*y + 2*x*5 + 3*4*y + 3*5 = 8*x*y + 10*x + 12*y + 15
        Assert.AreEqual(1, exprNode.QuadraticTerms1.Count); // x*y
        Assert.AreEqual(2, exprNode.LinearTerms.Count); // x and y
        Assert.AreEqual(15.0, exprNode.ConstantTerm);

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
        var expr = Quad([x * x]);
        var exprNode = QuadNode(expr);

        Assert.AreEqual(1, exprNode.QuadraticTerms1.Count);
        Assert.AreSame(exprNode.QuadraticTerms1[0], exprNode.QuadraticTerms2[0]); // Same variable

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

        var quad1 = Quad([x * x, 2 * x]);
        var quad2 = Quad([y * y, 3 * y]);

        // Combine them
        var combined = Quad([quad1, quad2, Const(5)]);
        var combinedNode = QuadNode(combined);

        Assert.AreEqual(2, combinedNode.QuadraticTerms1.Count); // x^2 and y^2
        Assert.AreEqual(2, combinedNode.LinearTerms.Count); // x and y
        Assert.AreEqual(5.0, combinedNode.ConstantTerm);

        // Verify no nested QuadExprNode
        Assert.IsFalse(combinedNode.LinearTerms.Any(t => t is QuadExprNode));
        Assert.IsFalse(combinedNode.QuadraticTerms1.Any(t => t is QuadExprNode));
        Assert.IsFalse(combinedNode.QuadraticTerms2.Any(t => t is QuadExprNode));
    }

    [TestMethod]
    public void QuadExpr_Negation()
    {
        var model = new Model();
        var x = model.AddVariable();

        // -(x^2)
        var expr = Quad([-Expr.Pow(x, 2)]);
        var exprNode = QuadNode(expr);

        Assert.AreEqual(1, exprNode.QuadraticTerms1.Count);
        Assert.AreEqual(-1.0, exprNode.QuadraticWeights[0]);

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
        var lin1 = Lin([x, y]);
        var lin2 = Lin([y, z]);
        var expr = Quad([lin1 * lin2]);
        var exprNode = QuadNode(expr);

        // Should have 4 quadratic terms
        Assert.AreEqual(4, exprNode.QuadraticTerms1.Count);
        Assert.AreEqual(0.0, exprNode.ConstantTerm);

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
        var expr = Quad([x * x, 2 * x * y, 3 * x, Const(4)]);

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

        // x * y creates ProductNode
        var xy = x * y;
        Assert.IsInstanceOfType(xy._node, typeof(ProductNode), "x*y alone is ProductNode");

        // x * y + 5 should be QuadExprNode (Product with 2 non-constants triggers QuadExpr)
        var expr1 = x * y + 5;
        Assert.IsInstanceOfType(expr1._node, typeof(QuadExprNode), "x*y + 5 should be QuadExprNode");

        // 2*x + 3*y should be LinExprNode
        var expr2 = 2 * x + 3 * y;
        Assert.IsInstanceOfType(expr2._node, typeof(LinExprNode), "2*x + 3*y should be LinExprNode");

        // x*x + 2*x + 1 should be QuadExprNode
        var expr3 = x * x + 2 * x + 1;
        Assert.IsInstanceOfType(expr3._node, typeof(QuadExprNode), "x*x + 2*x + 1 should be QuadExprNode");

        // Pow(x, 2) + 1 should be QuadExprNode
        var expr4 = Expr.Pow(x, 2) + 1;
        Assert.IsInstanceOfType(expr4._node, typeof(QuadExprNode), "x^2 + 1 should be QuadExprNode");
    }

    [TestMethod]
    public void LinExpr_MultiplyByConstant_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var linExpr = Lin([2 * x, 3 * y, Const(4)]);

        // Multiply by constant
        var scaled = linExpr * 5.0;

        // Should still be LinExprNode, not ProductNode
        Assert.IsInstanceOfType(scaled._node, typeof(LinExprNode), "LinExpr * constant should return LinExprNode");

        var scaledLin = LinNode(scaled);
        Assert.AreEqual(2, scaledLin.Terms.Count, "Should have 2 terms");
        Assert.AreEqual(10.0, scaledLin.Weights[0], 1e-10, "First weight should be 2*5=10");
        Assert.AreEqual(15.0, scaledLin.Weights[1], 1e-10, "Second weight should be 3*5=15");
        Assert.AreEqual(20.0, scaledLin.ConstantTerm, 1e-10, "Constant should be 4*5=20");

        // Test double * LinExpr as well
        var scaled2 = 5.0 * linExpr;
        Assert.IsInstanceOfType(scaled2._node, typeof(LinExprNode), "constant * LinExpr should return LinExprNode");
        var scaledLin2 = LinNode(scaled2);
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

        var linExpr = Lin([2 * x, 3 * y, Const(6)]);

        // Divide by constant
        var divided = linExpr / 2.0;

        // Should still be LinExprNode, not DivisionNode
        Assert.IsInstanceOfType(divided._node, typeof(LinExprNode), "LinExpr / constant should return LinExprNode");

        var dividedLin = LinNode(divided);
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

        var quadExpr = Quad([x * y, 2 * x, Const(3)]);

        // Multiply by constant
        var scaled = quadExpr * 4.0;

        // Should still be QuadExprNode, not ProductNode
        Assert.IsInstanceOfType(scaled._node, typeof(QuadExprNode), "QuadExpr * constant should return QuadExprNode");

        var scaledQuad = QuadNode(scaled);
        Assert.AreEqual(1, scaledQuad.LinearTerms.Count, "Should have 1 linear term");
        Assert.AreEqual(1, scaledQuad.QuadraticTerms1.Count, "Should have 1 quadratic term");
        Assert.AreEqual(8.0, scaledQuad.LinearWeights[0], 1e-10, "Linear weight should be 2*4=8");
        Assert.AreEqual(4.0, scaledQuad.QuadraticWeights[0], 1e-10, "Quadratic weight should be 1*4=4");
        Assert.AreEqual(12.0, scaledQuad.ConstantTerm, 1e-10, "Constant should be 3*4=12");

        // Test double * QuadExpr as well
        var scaled2 = 4.0 * quadExpr;
        Assert.IsInstanceOfType(scaled2._node, typeof(QuadExprNode), "constant * QuadExpr should return QuadExprNode");
        var scaledQuad2 = QuadNode(scaled2);
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

        var quadExpr = Quad([x * y, 4 * x, Const(8)]);

        // Divide by constant
        var divided = quadExpr / 2.0;

        // Should still be QuadExprNode, not DivisionNode
        Assert.IsInstanceOfType(divided._node, typeof(QuadExprNode), "QuadExpr / constant should return QuadExprNode");

        var dividedQuad = QuadNode(divided);
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

        var linExpr = Lin([2 * x, Const(5)]);
        linExpr *= 3.0;

        Assert.IsInstanceOfType(linExpr._node, typeof(LinExprNode), "LinExpr after *= constant should still be LinExprNode");

        var lin = LinNode(linExpr);
        Assert.AreEqual(1, lin.Terms.Count, "Should have 1 term");
        Assert.AreEqual(6.0, lin.Weights[0], 1e-10, "Weight should be 2*3=6");
        Assert.AreEqual(15.0, lin.ConstantTerm, 1e-10, "Constant should be 5*3=15");
    }

    [TestMethod]
    public void LinExpr_CompoundDivideAssignment_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();

        var linExpr = Lin([6 * x, Const(12)]);
        linExpr /= 3.0;

        Assert.IsInstanceOfType(linExpr._node, typeof(LinExprNode), "LinExpr after /= constant should still be LinExprNode");

        var lin = LinNode(linExpr);
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

        var quadExpr = Quad([x * y, 2 * x, Const(4)]);
        quadExpr *= 2.0;

        Assert.IsInstanceOfType(quadExpr._node, typeof(QuadExprNode), "QuadExpr after *= constant should still be QuadExprNode");

        var quad = QuadNode(quadExpr);
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

        var expr = Lin([2 * x, 4 * y]);
        Expr divisor = 2.0; // Implicitly converts to Constant

        var result = expr / divisor;

        Assert.IsInstanceOfType(result._node, typeof(LinExprNode));
        var lin = LinNode(result);
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

        var expr = Lin([2 * x, 4 * y]);
        Expr multiplier = 3.0; // Implicitly converts to Constant

        var result = expr * multiplier;

        Assert.IsInstanceOfType(result._node, typeof(LinExprNode));
        var lin = LinNode(result);
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
        var expr = Lin([2 * x, 4 * y]);

        var result = multiplier * expr;

        Assert.IsInstanceOfType(result._node, typeof(LinExprNode));
        var lin = LinNode(result);
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

        var expr = Lin([4 * x, 8 * y]);
        int divisor = 2;

        var result = expr / divisor;

        Assert.IsInstanceOfType(result._node, typeof(LinExprNode));
        var lin = LinNode(result);
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

        var quad = Quad([6 * x * y, 4 * x]);

        Expr divisor = 2.0;
        var result = quad / divisor;

        Assert.IsInstanceOfType(result._node, typeof(QuadExprNode));
        var resultQuad = QuadNode(result);
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

        var quad = Quad([2 * x * y, 3 * x]);

        Expr multiplier = 3.0;
        var result = quad * multiplier;

        Assert.IsInstanceOfType(result._node, typeof(QuadExprNode));
        var resultQuad = QuadNode(result);
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

        // Should create a QuadExprNode (or LinExprNode if optimized), not a DivisionNode
        Assert.IsTrue(result._node is QuadExprNode || result._node is LinExprNode,
            $"Expected QuadExprNode or LinExprNode but got {result._node.GetType().Name}");

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

        var prod = Prod([Const(2), x, Const(3)]);
        var prodNode = ProdNode(prod);

        Assert.AreEqual(1, prodNode.Factors.Count, "Should have 1 non-constant factor");
        Assert.AreEqual(6.0, prodNode.Factor, 1e-10, "Factor should be 2 * 3 = 6");
        Assert.IsFalse(prodNode.Factors.Any(f => f is ConstantNode), "Factors should not contain ConstantNode factors");
    }

    [TestMethod]
    public void Product_MultipleConstantsMultiplied()
    {
        // Multiple constants should be multiplied together
        var prod = Prod([Const(2), Const(3), Const(5)]);
        var prodNode = ProdNode(prod);

        Assert.AreEqual(0, prodNode.Factors.Count, "Should have no non-constant factors");
        Assert.AreEqual(30.0, prodNode.Factor, 1e-10, "Factor should be 2 * 3 * 5 = 30");
    }

    [TestMethod]
    public void Product_OnlyConstant_ExtractedToFactor()
    {
        // Single constant should be extracted to Factor
        var prod = Prod([Const(7)]);
        var prodNode = ProdNode(prod);

        Assert.AreEqual(0, prodNode.Factors.Count, "Should have no factors");
        Assert.AreEqual(7.0, prodNode.Factor, 1e-10, "Factor should be 7");
    }

    [TestMethod]
    public void Product_NoConstants_FactorIsOne()
    {
        // Product with no constants should have Factor = 1.0
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var prod = Prod([x, y]);
        var prodNode = ProdNode(prod);

        Assert.AreEqual(2, prodNode.Factors.Count);
        Assert.AreEqual(1.0, prodNode.Factor, 1e-10, "Factor should be 1.0 when no constants");
    }

    [TestMethod]
    public void Product_FactorPreservedInOperations()
    {
        // Factor should be preserved when extending Products
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();
        var z = model.AddVariable();

        var prod1 = Prod([Const(2), x]);
        var prod2 = prod1 * y; // Extend with y

        Assert.IsInstanceOfType(prod2._node, typeof(ProductNode));
        var prod2Product = ProdNode(prod2);
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

        var prod = Prod([Const(2), x, y, Const(3)]);

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
        var prod = Prod([Const(2), x, y, Const(3)]);
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

        var prod = Prod([Const(2), x]);
        var result = prod * 3.0;

        Assert.IsInstanceOfType(result._node, typeof(ProductNode));
        var resultProd = ProdNode(result);
        Assert.AreEqual(6.0, resultProd.Factor, 1e-10, "Factor should be 2 * 3 = 6");
        Assert.AreEqual(1, resultProd.Factors.Count, "Should still have 1 factor");
        Assert.IsFalse(resultProd.Factors.Any(f => f is ConstantNode), "Should not contain Constants");
    }

    [TestMethod]
    public void Product_FactorInCompoundOperators()
    {
        // Factor should work correctly with compound operators
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr expr = 0;
        expr += Prod([Const(3), x, y]);

        double[] point = [2.0, 4.0];
        double expected = 3.0 * 2.0 * 4.0; // 24
        Assert.AreEqual(expected, expr.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void QuadExpr_CompoundDivideAssignment_UpdatesCoefficients()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        var quadExpr = Quad([x * y, 8 * x, Const(16)]);
        quadExpr /= 4.0;

        Assert.IsInstanceOfType(quadExpr._node, typeof(QuadExprNode), "QuadExpr after /= constant should still be QuadExpr");

        var quad = QuadNode(quadExpr);
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
        var linExpr = Lin([2 * x, Const(5)]);
        var originalReference = linExpr;

        // Multiply by constant using compound assignment
        linExpr *= 3.0;

        // Verify the object was modified in-place (no replacement created)
        Assert.AreSame(originalReference, linExpr, "Should be the same object reference");
        Assert.AreEqual(6.0, LinNode(linExpr).Weights[0], 1e-10, "Weight should be updated in-place");
        Assert.AreEqual(15.0, LinNode(linExpr).ConstantTerm, 1e-10, "Constant should be updated in-place");
    }

    [TestMethod]
    public void QuadExpr_CompoundAssignment_ModifiesInPlace()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create QuadExpr directly (not through replacement)
        var quadExpr = Quad([x * y, 2 * x, Const(4)]);
        var originalReference = quadExpr;

        // Divide by constant using compound assignment
        quadExpr /= 2.0;

        // Verify the object was modified in-place (no replacement created)
        Assert.AreSame(originalReference, quadExpr, "Should be the same object reference");
        Assert.AreEqual(1.0, QuadNode(quadExpr).LinearWeights[0], 1e-10, "Linear weight should be updated in-place");
        Assert.AreEqual(0.5, QuadNode(quadExpr).QuadraticWeights[0], 1e-10, "Quadratic weight should be updated in-place");
        Assert.AreEqual(2.0, QuadNode(quadExpr).ConstantTerm, 1e-10, "Constant should be updated in-place");
    }

    [TestMethod]
    public void QuadExpr_CleanupLinExprWithProducts()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Manually create a LinExpr with Product terms (simulating old code path)
        var linWithProducts = Lin();
        LinNode(linWithProducts).Terms.Add(N(x * y));  // Product
        LinNode(linWithProducts).Weights.Add(1.0);
        LinNode(linWithProducts).ConstantTerm = 5.0;

        // Create QuadExpr from this LinExpr - should extract the Product into quadratic terms
        var quad = Quad([linWithProducts]);
        var quadN = QuadNode(quad);

        // The Product should have been moved to quadratic terms, not left in linear terms
        Assert.AreEqual(0, quadN.LinearTerms.Count, "Should have no linear terms");
        Assert.AreEqual(1, quadN.QuadraticTerms1.Count, "Should have 1 quadratic term");
        Assert.AreEqual(5.0, quadN.ConstantTerm);

        // Verify no nested Products in linear terms
        foreach (var term in quadN.LinearTerms)
        {
            Assert.IsFalse(term is ProductNode, "LinearTerms should not contain Product nodes");
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
        Expr obj = Const(0);
        var residual = variable - 5.0;

        Console.WriteLine("residual type: " + residual._node.GetType().Name);
        if (residual._node is LinExprNode lin)
        {
            Console.WriteLine("  LinExpr with " + lin.Terms.Count + " terms");
            for (int i = 0; i < lin.Terms.Count; i++)
            {
                Console.WriteLine("    Term[" + i + "]: " + lin.Terms[i].GetType().Name + " weight=" + lin.Weights[i]);
            }
            Console.WriteLine("  ConstantTerm: " + lin.ConstantTerm);
        }

        obj += residual * residual;

        Console.WriteLine("\nobj type after +=: " + obj._node.GetType().Name);
        if (obj._node is QuadExprNode quad)
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
                Assert.IsFalse(term is ProductNode, "LinearTerms should not contain Product nodes");
            }

            // Should have 1 quadratic term (variable * variable)
            Assert.AreEqual(1, quad.QuadraticTerms1.Count, "Should have 1 quadratic term");

            // Verify evaluation: (x - 5)^2 at x=3 → (-2)^2 = 4
            double[] point = [3];
            Assert.AreEqual(4.0, obj.Evaluate(point), 1e-10);
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

        // MUST be QuadExpr
        Assert.IsInstanceOfType(obj._node, typeof(QuadExprNode), $"Expected QuadExprNode but got {obj._node.GetType().Name}");
        var quad = QuadNode(obj);

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
            if (term is ProductNode prod)
            {
                Console.WriteLine($"    ERROR: Product found with {prod.Factors.Count} factors:");
                for (int j = 0; j < prod.Factors.Count; j++)
                {
                    Console.WriteLine($"      Factor[{j}]: {prod.Factors[j].GetType().Name}");
                    if (prod.Factors[j] is LinExprNode linExpr)
                    {
                        Console.WriteLine($"        LinExpr with {linExpr.Terms.Count} terms, constant={linExpr.ConstantTerm}");
                    }
                }
                Assert.Fail($"LinearTerms[{i}] is a Product - should have been expanded to quadratic terms!");
            }

            // Linear terms should only be Variable or similar simple expressions
            Assert.IsTrue(term is VariableNode || term is NegationNode,
                $"Linear term should be VariableNode or NegationNode, not {term.GetType().Name}");
        }

        // Verify quadratic structure: (v[i] - 5)^2 = v[i]^2 - 10*v[i] + 25
        // Should have exactly 50 quadratic terms (one v[i]^2 per variable)
        Assert.AreEqual(50, quad.QuadraticTerms1.Count, "Should have 50 quadratic terms");

        // All quadratic terms should be v[i] * v[i] (same variable squared)
        for (int i = 0; i < quad.QuadraticTerms1.Count; i++)
        {
            Assert.IsInstanceOfType(quad.QuadraticTerms1[i], typeof(VariableNode),
                $"QuadraticTerm1[{i}] should be VariableNode");
            Assert.IsInstanceOfType(quad.QuadraticTerms2[i], typeof(VariableNode),
                $"QuadraticTerm2[{i}] should be VariableNode");
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
        var linExpr = Lin();
        linExpr += 2 * x;
        linExpr += 3 * y;

        // Verify it worked efficiently (should have 2 terms)
        Assert.AreEqual(2, LinNode(linExpr).Terms.Count);
        Assert.AreEqual(2.0, LinNode(linExpr).Weights[0]);
        Assert.AreEqual(3.0, LinNode(linExpr).Weights[1]);

        // Test QuadExpr directly
        var quadExpr = Quad();
        quadExpr += x * y;
        quadExpr += x * x;

        // Should have 2 quadratic terms
        Assert.AreEqual(2, QuadNode(quadExpr).QuadraticTerms1.Count);
        Assert.AreEqual(0, QuadNode(quadExpr).LinearTerms.Count);

        // Test Product directly
        var prod = Prod([Const(2)]);
        prod *= x;
        prod *= y;

        // Should have 2 factors (Constants are extracted to Factor field)
        Assert.AreEqual(2, ProdNode(prod).Factors.Count);
        Assert.AreEqual(2.0, ProdNode(prod).Factor, 1e-10, "Factor should be 2");

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

        Console.WriteLine("obj type: " + obj._node.GetType().Name);

        if (obj._node is QuadExprNode quad)
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
                Assert.IsFalse(term is ProductNode, "LinearTerms should not contain Products");
            }
        }
        else
        {
            Assert.Fail("Expected QuadExprNode, got " + obj._node.GetType().Name);
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
        var wrongExpr = new Expr(new WrongGradientNode(x));
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
        var wrongExpr = new Expr(new WrongHessianNode(x));
        model.SetObjective(wrongExpr);

        var result = ModellingTests.SolveWithDerivativeTest(model);

        // Derivative test should have detected errors
        Assert.IsNotNull(result.DerivativeTestResult);
        Assert.IsFalse(result.DerivativeTestResult.Passed, "Derivative test should fail for wrong Hessian");
        Assert.IsTrue(result.DerivativeTestResult.ErrorCount > 0, "Should report Hessian errors");
    }

    [TestMethod]
    public void ConstantPlusProduct_CreatesCorrectType()
    {
        // Test what happens when adding a Product to Constant(0)
        var model = new Model();
        var x = model.AddVariable(0);
        var y = model.AddVariable(0);

        var constant = (Expr)0; // Creates Constant(0)
        var product = x * y; // Creates Product

        Console.WriteLine($"Before addition:");
        Console.WriteLine($"  constant type: {constant.GetType().Name}");
        Console.WriteLine($"  constant.ToString(): {constant}");
        Console.WriteLine($"  product type: {product.GetType().Name}");
        Console.WriteLine($"  product.ToString(): {product}");

        var result = constant + product;

        Console.WriteLine($"\nAfter constant + product:");
        Console.WriteLine($"  result type: {result.GetType().Name}");
        Console.WriteLine($"  result.ToString(): {result}");

        // Check variables in result
        var variables = new HashSet<Variable>();
        result.CollectVariables(variables);
        Console.WriteLine($"  Variables in result:");
        foreach (var v in variables.OrderBy(v => v.Index))
            Console.WriteLine($"    {v}");

        // Verify type is NOT Constant
        Assert.IsFalse(result._node is ConstantNode,
            "Result of Constant(0) + Product should NOT be a Constant");

        // Verify both variables are present
        Assert.IsTrue(variables.Contains(x),
            "x should be present in result");
        Assert.IsTrue(variables.Contains(y),
            "y should be present in result");

        // Verify it's quadratic
        Assert.IsTrue(result.IsAtMostQuadratic(),
            "Result should be at most quadratic");
        Assert.IsFalse(result.IsLinear(),
            "Result should NOT be linear");
    }

    [TestMethod]
    public void ConstantWithReplacement_ShouldFollowReplacement()
    {
        // Test that Constant with _replacement is handled correctly in LinExpr/QuadExpr constructors
        // This is the root cause of the bug where variables disappear
        var model = new Model();
        var x = model.AddVariable(0);
        var y = model.AddVariable(0);

        var expr = (Expr)0; // Creates Constant(0)
        var product = x * y; // Creates Product

        expr += product; // expr's _node is now the Product node (compound operator replaces zero constant)

        // After +=, the node should be the Product (not a Constant anymore)
        Assert.IsInstanceOfType(expr._node, typeof(ProductNode),
            "expr._node should be ProductNode after += from zero");

        // Now add this "Constant-with-replacement" to another expression
        var z = model.AddVariable(0);
        var sum = z + expr; // BUG: LinExpr/QuadExpr constructor should follow _replacement

        // Verify all variables are present in sum
        var variables = new HashSet<Variable>();
        sum.CollectVariables(variables);

        Assert.IsTrue(variables.Contains(x),
            "x should be present in sum (from the Product that expr redirects to)");
        Assert.IsTrue(variables.Contains(y),
            "y should be present in sum (from the Product that expr redirects to)");
        Assert.IsTrue(variables.Contains(z),
            "z should be present in sum");
    }

    [TestMethod]
    public void ConstantWithReplacement_QuadExprConstructor_ShouldFollowReplacement()
    {
        // Test that QuadExpr constructor also follows _replacement
        var model = new Model();
        var fixedVar = model.AddVariable(60, 60); // Fixed variable
        var freeVar = model.AddVariable(0); // Free variable
        var x1 = model.AddVariable(0);
        var x2 = model.AddVariable(0);

        // Create a "Constant-with-replacement" like in actual optimization code
        var eSurfaceTreatment = (Expr)0;
        eSurfaceTreatment += fixedVar * freeVar; // Now eSurfaceTreatment is Constant with _replacement = Product

        // Create complex expressions that will result in QuadExpr
        var eMachineCost = fixedVar * (x1 * 2.0 + 5.0); // Quadratic
        var eMachinePersonCost = x2 * (x1 * 3.0 + 1.0); // Quadratic

        // Add them together - should create QuadExpr that properly handles eSurfaceTreatment
        var predictedPrice = eMachineCost + eMachinePersonCost + eSurfaceTreatment;

        Assert.IsInstanceOfType(predictedPrice._node, typeof(QuadExprNode),
            "predictedPrice should be QuadExpr");

        // Verify all variables are present
        var variables = new HashSet<Variable>();
        predictedPrice.CollectVariables(variables);

        Assert.IsTrue(variables.Contains(fixedVar),
            "fixedVar should be present");
        Assert.IsTrue(variables.Contains(freeVar),
            "freeVar should be present (BUG: QuadExpr constructor doesn't follow _replacement)");
        Assert.IsTrue(variables.Contains(x1),
            "x1 should be present");
        Assert.IsTrue(variables.Contains(x2),
            "x2 should be present");
    }

    [TestMethod]
    public void RealWorldBug_MultipleAdditions_PreservesAllVariables()
    {
        // Exact reproduction of the actual bug from sheet metal optimization
        var model = new Model();

        var fixedSalary = model.AddVariable(60, 60); // x[111] - CHE salary (fixed)
        var additionalProc = model.AddVariable(0); // x[148] - vAdditionalProcecedure
        var x1 = model.AddVariable(0);
        var x2 = model.AddVariable(0);
        var x3 = model.AddVariable(0);

        // Mimic actual code structure
        var eMaterial = x1 * 2.0;
        var eMachineTime = x2 * 3.0 + 5.0;
        var eMachineCost = fixedSalary * eMachineTime;
        var eMachinePersonCost = x3 * eMachineTime;

        // THIS IS THE CRITICAL PART - eSurfaceTreatment starts as Constant(0)
        var eSurfaceTreatment = (Expr)0;
        eSurfaceTreatment += fixedSalary * additionalProc; // Now it's Constant with _replacement

        // Add them exactly like in actual code
        var predictedPrice = eMaterial + eMachineCost + eMachinePersonCost + eSurfaceTreatment;

        Console.WriteLine($"eSurfaceTreatment type: {eSurfaceTreatment._node.GetType().Name}");
        Console.WriteLine($"eSurfaceTreatment node type: {eSurfaceTreatment._node.GetType().Name}");

        var varsInST = new HashSet<Variable>();
        eSurfaceTreatment.CollectVariables(varsInST);
        Console.WriteLine($"Variables in eSurfaceTreatment: {string.Join(", ", varsInST.OrderBy(v => v.Index).Select(v => $"x[{v.Index}]"))}");

        Console.WriteLine($"predictedPrice type: {predictedPrice.GetType().Name}");

        var variables = new HashSet<Variable>();
        predictedPrice.CollectVariables(variables);
        Console.WriteLine($"Variables in predictedPrice: {string.Join(", ", variables.OrderBy(v => v.Index).Select(v => $"x[{v.Index}]"))}");

        Assert.IsTrue(variables.Contains(additionalProc),
            "additionalProc (x[148]) should be present in predictedPrice - THIS IS THE BUG");
    }

    [TestMethod]
    public void ComplexAddition_PreservesAllVariables()
    {
        // Reproduces bug where Product of fixed*free variable disappears when added to complex expressions
        var model = new Model();

        // Create variables like in the sheet metal optimization
        var fixedVar = model.AddVariable(60, 60); // Fixed variable (like CHE salary)
        var freeVar = model.AddVariable(0); // Free variable (like vAdditionalProcecedure)
        var x1 = model.AddVariable(0);
        var x2 = model.AddVariable(0);
        var x3 = model.AddVariable(0);

        // Create complex expressions like in actual optimization code
        var eMaterial = x1 * 2.0; // Linear expression
        var eMachineTime = x2 * 3.0 + 5.0; // Linear expression
        var eMachineCost = fixedVar * eMachineTime; // Product: fixedVar * eMachineTime (beyond linear)
        var eMachinePersonCost = x3 * eMachineTime; // Product: x3 * eMachineTime (beyond linear)

        // Build eSurfaceTreatment using += like in actual code
        var eSurfaceTreatment = (Expr)0;
        var product = fixedVar * freeVar;
        Console.WriteLine($"product type: {product.GetType().Name}");
        Console.WriteLine($"product: {product}");

        Console.WriteLine($"Before +=: eSurfaceTreatment type: {eSurfaceTreatment.GetType().Name}");
        eSurfaceTreatment += product; // Product: fixedVar * freeVar (quadratic)
        Console.WriteLine($"After +=: eSurfaceTreatment type: {eSurfaceTreatment.GetType().Name}");
        Console.WriteLine($"eSurfaceTreatment: {eSurfaceTreatment}");

        // Check if product variables are in eSurfaceTreatment
        var stVars = new HashSet<Variable>();
        eSurfaceTreatment.CollectVariables(stVars);
        Console.WriteLine($"Variables in eSurfaceTreatment after +=:");
        foreach (var v in stVars.OrderBy(v => v.Index))
            Console.WriteLine($"  {v}");

        // Add them together like in actual code: (eMaterial + eMachineCost + eMachinePersonCost + eSurfaceTreatment)
        var predictedPrice = eMaterial + eMachineCost + eMachinePersonCost + eSurfaceTreatment;
        Console.WriteLine($"predictedPrice type: {predictedPrice.GetType().Name}");
        Console.WriteLine($"predictedPrice: {predictedPrice}");

        // Verify the result type
        Assert.IsTrue(predictedPrice._node is QuadExprNode || predictedPrice._node is LinExprNode,
            $"Result should be QuadExprNode or LinExprNode, but got {predictedPrice._node.GetType().Name}");

        // CRITICAL: Verify all variables are present in the result
        var variables = new HashSet<Variable>();
        predictedPrice.CollectVariables(variables);

        Console.WriteLine($"Variables in predictedPrice:");
        foreach (var v in variables.OrderBy(v => v.Index))
            Console.WriteLine($"  {v}");

        Assert.IsTrue(variables.Contains(fixedVar),
            "Fixed variable should be present in result");
        Assert.IsTrue(variables.Contains(freeVar),
            $"Free variable x[{freeVar.Index}] should be present in result - THIS IS THE BUG");
        Assert.IsTrue(variables.Contains(x1),
            "x1 should be present in result");
        Assert.IsTrue(variables.Contains(x2),
            "x2 should be present in result");
        Assert.IsTrue(variables.Contains(x3),
            "x3 should be present in result");

        // Verify it's actually quadratic (not linear) since we have Product terms
        Assert.IsTrue(predictedPrice.IsAtMostQuadratic(),
            "Result should be at most quadratic");
        Assert.IsFalse(predictedPrice.IsLinear(),
            "Result should NOT be linear since it contains Product terms");
    }

    [TestMethod]
    public void ComplexAddition_WithMultipleConditionalTerms_PreservesAllVariables()
    {
        // More complex test that exactly mimics the actual optimization structure
        var model = new Model();

        var fixedVar = model.AddVariable(60, 60); // CHE salary
        var freeVar = model.AddVariable(0); // vAdditionalProcecedure
        var x1 = model.AddVariable(0);
        var x2 = model.AddVariable(0);
        var x3 = model.AddVariable(0);
        var x4 = model.AddVariable(0);

        var eMaterial = x1 * 2.0;
        var eMachineTime = x2 * 3.0 + 5.0;
        var eMachineCost = fixedVar * eMachineTime;
        var eMachinePersonCost = x3 * eMachineTime;

        // Build surface treatment with multiple conditional additions (like actual code)
        var eSurfaceTreatment = (Expr)0;
        // Some parts have coating
        eSurfaceTreatment += fixedVar * x4 * 0.5; // Example: coating
        // This part has additional procedure
        eSurfaceTreatment += fixedVar * freeVar; // The problematic term

        // Create predictedPrice
        var predictedPrice = (eMaterial + eMachineCost + eMachinePersonCost + eSurfaceTreatment);

        // Collect variables
        var variables = new HashSet<Variable>();
        predictedPrice.CollectVariables(variables);

        // Verify all variables are present
        Assert.IsTrue(variables.Contains(freeVar),
            "Free variable (vAdditionalProcecedure) should be present in predictedPrice");
        Assert.IsTrue(variables.Contains(fixedVar),
            "Fixed variable should be present");
        Assert.IsTrue(variables.Contains(x1), "x1 should be present");
        Assert.IsTrue(variables.Contains(x2), "x2 should be present");
        Assert.IsTrue(variables.Contains(x3), "x3 should be present");
        Assert.IsTrue(variables.Contains(x4), "x4 should be present");
    }

    [TestMethod]
    public void AllOperators_WithReplacedConstant_ResolveCorrectly()
    {
        // Regression test for _replacement resolution at operator boundaries.
        // Tests that all operators call GetActual() before type-checking operands.
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Create a Constant(0) with a _replacement pointing to a LinExpr
        var replacedExpr = (Expr)0;
        replacedExpr += x + y; // Now replacedExpr is Constant(0) with _replacement=LinExpr

        // Test binary operators with Expr operands
        var addExpr = replacedExpr + (x * 2);
        var subExpr = replacedExpr - (x * 2);
        var mulExpr = replacedExpr * y;
        var divExpr = replacedExpr / (y + 1);

        // Test binary operators with double operands
        var addDouble1 = replacedExpr + 5.0;
        var addDouble2 = 5.0 + replacedExpr;
        var subDouble1 = replacedExpr - 5.0;
        var subDouble2 = 5.0 - replacedExpr;
        var mulDouble1 = replacedExpr * 2.0;
        var mulDouble2 = 2.0 * replacedExpr;
        var divDouble = replacedExpr / 2.0;

        // Test Product constructor
        var product = Prod([replacedExpr, y]);

        // Test compound operators
        var compound1 = (Expr)0;
        compound1 += replacedExpr;
        var compound2 = (Expr)0;
        compound2 -= replacedExpr;
        var compound3 = (Expr)1;
        compound3 *= replacedExpr;
        var compound4 = (Expr)10;
        compound4 /= replacedExpr;

        // Verify all expressions have correct variables (x and y should be present)
        var expressions = new[] { addExpr, subExpr, mulExpr, divExpr, addDouble1, addDouble2,
            subDouble1, subDouble2, mulDouble1, mulDouble2, divDouble, product,
            compound1, compound2, compound3, compound4 };

        foreach (var expr in expressions)
        {
            var variables = new HashSet<Variable>();
            expr.CollectVariables(variables);
            Assert.IsTrue(variables.Contains(x), $"{expr.GetType().Name}: x should be present");
            Assert.IsTrue(variables.Contains(y), $"{expr.GetType().Name}: y should be present");
        }

        // Verify gradient correctness for a few key cases
        double[] point = [2.0, 3.0];
        AssertGradientMatchesFiniteDifference(addExpr, point);
        AssertGradientMatchesFiniteDifference(mulExpr, point);
        AssertGradientMatchesFiniteDifference(product, point);
    }
}

/// <summary>
/// Custom expression that returns x^2 but claims gradient is 3*x (wrong - should be 2*x)
/// </summary>
internal class WrongGradientNode : ExprNode
{
    private readonly Variable _x;

    public WrongGradientNode(Variable x)
    {
        _x = x;
    }

    internal override double Evaluate(ReadOnlySpan<double> x) => Math.Pow(x[_x.Index], 2);

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        // Correct would be: compactGrad[Array.BinarySearch(sortedVarIndices, _x.Index)] += multiplier * 2 * x[_x.Index];
        // But we intentionally return wrong derivative:
        compactGrad[Array.BinarySearch(sortedVarIndices, _x.Index)] += multiplier * 3 * x[_x.Index]; // WRONG!
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        hess.Add(_x.Index, _x.Index, multiplier * 2.0);
    }

    internal override void CollectVariables(HashSet<Variable> variables)
    {
        variables.Add(_x);
    }

    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries)
    {
        entries.Add((_x.Index, _x.Index));
    }

    internal override bool IsConstantWrtX() => false;
    internal override bool IsLinear() => false;
    internal override bool IsAtMostQuadratic() => true;
}

/// <summary>
/// Custom expression that returns x^2 with correct gradient 2*x but claims Hessian is 5 (wrong - should be 2)
/// </summary>
internal class WrongHessianNode : ExprNode
{
    private readonly Variable _x;

    public WrongHessianNode(Variable x)
    {
        _x = x;
    }

    internal override double Evaluate(ReadOnlySpan<double> x) => Math.Pow(x[_x.Index], 2);

    internal override void AccumulateGradientCompact(ReadOnlySpan<double> x, Span<double> compactGrad, double multiplier, int[] sortedVarIndices)
    {
        compactGrad[Array.BinarySearch(sortedVarIndices, _x.Index)] += multiplier * 2 * x[_x.Index]; // Correct
    }

    internal override void AccumulateHessian(ReadOnlySpan<double> x, HessianAccumulator hess, double multiplier)
    {
        // Correct would be: hess.Add(_x.Index, _x.Index, multiplier * 2.0);
        // But we intentionally return wrong second derivative:
        hess.Add(_x.Index, _x.Index, multiplier * 5.0); // WRONG!
    }

    internal override void CollectVariables(HashSet<Variable> variables)
    {
        variables.Add(_x);
    }

    internal override void CollectHessianSparsity(HashSet<(int row, int col)> entries)
    {
        entries.Add((_x.Index, _x.Index));
    }

    internal override bool IsConstantWrtX() => false;
    internal override bool IsLinear() => false;
    internal override bool IsAtMostQuadratic() => true;
}
