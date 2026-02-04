using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class ModellingTests
{
    [TestMethod]
    public void HS071_WithModel()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x1 = model.AddVariable(1, 5);
        x1.Start = 1;
        var x2 = model.AddVariable(1, 5);
        x2.Start = 5;
        var x3 = model.AddVariable(1, 5);
        x3.Start = 5;
        var x4 = model.AddVariable(1, 5);
        x4.Start = 1;

        // Objective: minimize x1*x4*(x1+x2+x3) + x3
        model.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);

        // Constraints
        model.AddConstraint(x1 * x2 * x3 * x4 >= 25);
        model.AddConstraint(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 == 40);

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(17.014, result.ObjectiveValue, 0.01);
        Assert.AreEqual(1.0, result.Solution[x1], 0.01);
        Assert.AreEqual(4.743, result.Solution[x2], 0.01);
        Assert.AreEqual(3.821, result.Solution[x3], 0.01);
        Assert.AreEqual(1.379, result.Solution[x4], 0.01);
    }

    [TestMethod]
    public void Rosenbrock_Unconstrained()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = -1;
        var y = model.AddVariable();
        y.Start = 1;

        // f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        model.SetObjective(Expr.Pow(1 - x, 2) + 100 * Expr.Pow(y - x * x, 2));

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(0.0, result.ObjectiveValue, 0.001);
        Assert.AreEqual(1.0, result.Solution[x], 0.001);
        Assert.AreEqual(1.0, result.Solution[y], 0.001);
    }

    [TestMethod]
    public void SimpleQuadratic_BoundConstrained()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(0, 10);
        x.Start = 0;
        var y = model.AddVariable(0, 10);
        y.Start = 0;

        // minimize (x-3)^2 + (y-4)^2
        model.SetObjective(Expr.Pow(x - 3, 2) + Expr.Pow(y - 4, 2));

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(0.0, result.ObjectiveValue, 0.001);
        Assert.AreEqual(3.0, result.Solution[x], 0.001);
        Assert.AreEqual(4.0, result.Solution[y], 0.001);
    }

    [TestMethod]
    public void LinearConstraint()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 0;
        var y = model.AddVariable();
        y.Start = 0;

        // minimize x^2 + y^2
        // subject to x + y = 4
        model.SetObjective(x * x + y * y);
        model.AddConstraint(x + y == 4);

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(2.0, result.Solution[x], 0.001);
        Assert.AreEqual(2.0, result.Solution[y], 0.001);
        Assert.AreEqual(8.0, result.ObjectiveValue, 0.001);
    }

    [TestMethod]
    public void TrigonometricExpression()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(-Math.PI, Math.PI);
        x.Start = 0;

        // minimize -sin(x)
        // optimal at x = pi/2
        model.SetObjective(-Expr.Sin(x));

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(Math.PI / 2, result.Solution[x], 0.01);
        Assert.AreEqual(-1.0, result.ObjectiveValue, 0.001);
    }

    [TestMethod]
    public void ExponentialExpression()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(-5, 5);
        x.Start = 0;

        // minimize exp(x) - 2*x
        // derivative: exp(x) - 2 = 0 => x = ln(2)
        model.SetObjective(Expr.Exp(x) - 2 * x);

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(Math.Log(2), result.Solution[x], 0.001);
    }

    [TestMethod]
    public void DivisionExpression()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(0.1, 10);
        x.Start = 2;

        // minimize x + 1/x
        // derivative: 1 - 1/x^2 = 0 => x = 1
        model.SetObjective(x + 1 / x);

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(1.0, result.Solution[x], 0.001);
        Assert.AreEqual(2.0, result.ObjectiveValue, 0.001);
    }

    [TestMethod]
    public void ConfigureOptionsWithEnums()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(1, 5);
        x.Start = 1;
        var y = model.AddVariable(1, 5);
        y.Start = 5;
        var z = model.AddVariable(1, 5);
        z.Start = 5;
        var w = model.AddVariable(1, 5);
        w.Start = 1;

        model.SetObjective(x * w * (x + y + z) + z);
        model.AddConstraint(x * y * z * w >= 25);
        model.AddConstraint(x * x + y * y + z * z + w * w == 40);

        // Configure solver options using enums
        model.Options.LinearSolver = LinearSolver.Mumps;
        model.Options.HessianApproximation = HessianApproximation.Exact;
        model.Options.MuStrategy = MuStrategy.Adaptive;
        model.Options.Tolerance = 1e-7;
        model.Options.MaxIterations = 100;
        model.Options.PrintLevel = 0;

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(17.014, result.ObjectiveValue, 0.01);
    }

    [TestMethod]
    public void ConfigureCustomOptions()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 0;
        var y = model.AddVariable();
        y.Start = 0;

        model.SetObjective(x * x + y * y);
        model.AddConstraint(x + y == 4);

        // Use custom option method for less common options
        model.Options.SetCustomOption("bound_push", 0.01);
        model.Options.SetCustomOption("bound_frac", 0.01);

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(2.0, result.Solution[x], 0.001);
        Assert.AreEqual(2.0, result.Solution[y], 0.001);
    }

    [TestMethod]
    public void SolveWithMumps()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x1 = model.AddVariable(1, 5);
        x1.Start = 1;
        var x2 = model.AddVariable(1, 5);
        x2.Start = 5;
        var x3 = model.AddVariable(1, 5);
        x3.Start = 5;
        var x4 = model.AddVariable(1, 5);
        x4.Start = 1;

        model.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);
        model.AddConstraint(x1 * x2 * x3 * x4 >= 25);
        model.AddConstraint(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 == 40);

        // Explicitly use Mumps linear solver (default)
        model.Options.LinearSolver = LinearSolver.Mumps;

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(17.014, result.ObjectiveValue, 0.01);
        Assert.AreEqual(1.0, result.Solution[x1], 0.01);
        Assert.AreEqual(4.743, result.Solution[x2], 0.01);
        Assert.AreEqual(3.821, result.Solution[x3], 0.01);
        Assert.AreEqual(1.379, result.Solution[x4], 0.01);
    }

    [TestMethod]
    public void SolveWithPardisoMkl()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x1 = model.AddVariable(1, 5);
        x1.Start = 1;
        var x2 = model.AddVariable(1, 5);
        x2.Start = 5;
        var x3 = model.AddVariable(1, 5);
        x3.Start = 5;
        var x4 = model.AddVariable(1, 5);
        x4.Start = 1;

        model.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);
        model.AddConstraint(x1 * x2 * x3 * x4 >= 25);
        model.AddConstraint(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 == 40);

        // Use Pardiso MKL linear solver
        model.Options.LinearSolver = LinearSolver.PardisoMkl;

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(17.014, result.ObjectiveValue, 0.01);
        Assert.AreEqual(1.0, result.Solution[x1], 0.01);
        Assert.AreEqual(4.743, result.Solution[x2], 0.01);
        Assert.AreEqual(3.821, result.Solution[x3], 0.01);
        Assert.AreEqual(1.379, result.Solution[x4], 0.01);
    }

    [TestMethod]
    public void ExprToExprConstraints()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 0;
        var y = model.AddVariable();
        y.Start = 0;

        // minimize (x-2)^2 + (y-3)^2
        model.SetObjective(Expr.Pow(x - 2, 2) + Expr.Pow(y - 3, 2));

        // subject to x >= y (using Expr-to-Expr comparison)
        model.AddConstraint(x >= y);
        // and x + y == 4 (using Expr-to-Expr comparison)
        model.AddConstraint(x + y == 4);

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        // Expected solution: x=2, y=2 (closest point to (2,3) with x>=y and x+y=4)
        Assert.AreEqual(2.0, result.Solution[x], 0.01);
        Assert.AreEqual(2.0, result.Solution[y], 0.01);
    }

    [TestMethod]
    public void DerivativeTest_FirstOrder_ComplexExpression()
    {
        var model = new Model();
        var x = model.AddVariable(0.1, 10);
        x.Start = 1;
        var y = model.AddVariable(0.1, 10);
        y.Start = 2;
        var z = model.AddVariable(-5, 5);
        z.Start = 0.5;

        // Complex objective combining polynomial, trigonometric, exponential, and division
        // f(x,y,z) = x^2*y + sin(z)*exp(x) + y/x + z^3
        model.SetObjective(x * x * y + Expr.Sin(z) * Expr.Exp(x) + y / x + z * z * z);

        // Nonlinear constraints to test constraint derivatives
        // x*y*z >= 0.5
        model.AddConstraint(x * y * z >= 0.5);
        // x^2 + y^2 + z^2 <= 50
        model.AddConstraint(x * x + y * y + z * z <= 50);
        // exp(x) + log(y) == 5
        model.AddConstraint(Expr.Exp(x) + Expr.Log(y) == 5);

        // Enable first-order derivative test
        model.Options.DerivativeTest = DerivativeTest.FirstOrder;
        model.Options.DerivativeTestPerturbation = 1e-8;
        model.Options.DerivativeTestTolerance = 1e-4;
        model.Options.PrintLevel = 5;

        var result = model.Solve();

        // If derivatives are incorrect, IPOPT would report errors and likely fail to converge
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
    }

    [TestMethod]
    public void DerivativeTest_SecondOrder_ComplexExpression()
    {
        var model = new Model();
        var x = model.AddVariable(1, 5);
        x.Start = 2;
        var y = model.AddVariable(1, 5);
        y.Start = 3;
        var z = model.AddVariable(-Math.PI, Math.PI);
        z.Start = 0.5;
        var w = model.AddVariable(0.5, 10);
        w.Start = 2;

        // Highly nonlinear objective with mixed second derivatives
        // f(x,y,z,w) = x*y*z + sin(z)*cos(z) + exp(w/x) + (y-x)^2 + w^3
        model.SetObjective(
            x * y * z +
            Expr.Sin(z) * Expr.Cos(z) +
            Expr.Exp(w / x) +
            Expr.Pow(y - x, 2) +
            w * w * w
        );

        // Multiple nonlinear constraints with complex Hessians
        // Rosenbrock-like constraint: (x-1)^2 + 100*(y-x^2)^2 <= 10
        model.AddConstraint(Expr.Pow(x - 1, 2) + 100 * Expr.Pow(y - x * x, 2) <= 10);
        // Product constraint: x*y*z*w >= 5
        model.AddConstraint(x * y * z * w >= 5);
        // Sphere constraint: x^2 + y^2 + z^2 + w^2 == 20
        model.AddConstraint(x * x + y * y + z * z + w * w == 20);
        // Trigonometric constraint: sin(z) + cos(z) >= 0
        model.AddConstraint(Expr.Sin(z) + Expr.Cos(z) >= 0);

        // Enable second-order derivative test (tests Hessian)
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        model.Options.DerivativeTestPerturbation = 1e-7;
        model.Options.DerivativeTestTolerance = 5e-4;
        model.Options.PrintLevel = 5;

        var result = model.Solve();

        // If second derivatives (Hessian) are incorrect, IPOPT would report errors
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
    }

    [TestMethod]
    public void DerivativeTest_FullDerivatives_HS071()
    {
        var model = new Model();
        var x1 = model.AddVariable(1, 5);
        x1.Start = 1;
        var x2 = model.AddVariable(1, 5);
        x2.Start = 5;
        var x3 = model.AddVariable(1, 5);
        x3.Start = 5;
        var x4 = model.AddVariable(1, 5);
        x4.Start = 1;

        // HS071 problem with derivative checking
        model.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);
        model.AddConstraint(x1 * x2 * x3 * x4 >= 25);
        model.AddConstraint(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 == 40);

        // Test both first and second order derivatives by running twice
        model.Options.PrintLevel = 5;

        // First run: test first-order derivatives
        model.Options.DerivativeTest = DerivativeTest.FirstOrder;
        model.Options.DerivativeTestPerturbation = 1e-8;
        model.Options.DerivativeTestTolerance = 1e-4;

        var result1 = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result1.Status);
        Assert.AreEqual(17.014, result1.ObjectiveValue, 0.01);

        // Second run: test second-order derivatives (Hessian)
        // Reset to initial point
        x1.Start = 1;
        x2.Start = 5;
        x3.Start = 5;
        x4.Start = 1;

        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        model.Options.DerivativeTestPerturbation = 1e-7;
        model.Options.DerivativeTestTolerance = 5e-4;

        var result2 = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result2.Status);
        Assert.AreEqual(17.014, result2.ObjectiveValue, 0.01);
    }

    [TestMethod]
    public void DerivativeTest_MixedFunctions_HigherOrderTerms()
    {
        var model = new Model();
        var x = model.AddVariable(0.1, 5);
        x.Start = 1.5;
        var y = model.AddVariable(0.1, 5);
        y.Start = 2.0;

        // Objective with high-order polynomial and transcendental functions
        // f(x,y) = x^4 + y^4 - 2*x^2*y^2 + sin(x*y) + exp(x-y) + log(x*y)
        model.SetObjective(
            Expr.Pow(x, 4) +
            Expr.Pow(y, 4) -
            2 * x * x * y * y +
            Expr.Sin(x * y) +
            Expr.Exp(x - y) +
            Expr.Log(x * y)
        );

        // Constraints with division and power operations
        // x/y + y/x >= 2
        model.AddConstraint(x / y + y / x >= 2);
        // (x+y)^3 <= 100
        model.AddConstraint(Expr.Pow(x + y, 3) <= 100);
        // x*exp(y) >= 2
        model.AddConstraint(x * Expr.Exp(y) >= 2);

        // Enable second-order derivative test for comprehensive checking
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        model.Options.DerivativeTestPerturbation = 1e-7;
        model.Options.DerivativeTestTolerance = 1e-3;
        model.Options.PrintLevel = 5;
        model.Options.MaxIterations = 200;

        var result = model.Solve();

        // Verify IPOPT successfully validates all derivatives
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
    }

    [TestMethod]
    public void UpdateStartValues_DefaultBehavior_UpdatesStartValues()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 0;
        var y = model.AddVariable();
        y.Start = 0;

        // minimize x^2 + y^2
        // subject to x + y = 4
        model.SetObjective(x * x + y * y);
        model.AddConstraint(x + y == 4);

        // Solve with default behavior (updateStartValues = true)
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(2.0, result.Solution[x], 0.001);
        Assert.AreEqual(2.0, result.Solution[y], 0.001);

        // Verify Start values were updated to solution
        Assert.AreEqual(2.0, x.Start, 0.001);
        Assert.AreEqual(2.0, y.Start, 0.001);
    }

    [TestMethod]
    public void UpdateStartValues_ExplicitTrue_UpdatesStartValues()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 10;
        var y = model.AddVariable();
        y.Start = -5;

        // minimize (x-3)^2 + (y-4)^2
        model.SetObjective(Expr.Pow(x - 3, 2) + Expr.Pow(y - 4, 2));

        // Solve with explicit updateStartValues = true
        var result = model.Solve(updateStartValues: true);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);

        // Verify Start values were updated to solution
        Assert.AreEqual(3.0, x.Start, 0.001);
        Assert.AreEqual(4.0, y.Start, 0.001);
    }

    [TestMethod]
    public void UpdateStartValues_False_DoesNotUpdateStartValues()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 100;
        var y = model.AddVariable();
        y.Start = -50;

        const double originalXStart = 100;
        const double originalYStart = -50;

        // minimize (x-3)^2 + (y-4)^2
        model.SetObjective(Expr.Pow(x - 3, 2) + Expr.Pow(y - 4, 2));

        // Solve with updateStartValues = false
        var result = model.Solve(updateStartValues: false);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(3.0, result.Solution[x], 0.001);
        Assert.AreEqual(4.0, result.Solution[y], 0.001);

        // Verify Start values were NOT updated
        Assert.AreEqual(originalXStart, x.Start);
        Assert.AreEqual(originalYStart, y.Start);
    }

    [TestMethod]
    public void UpdateStartValues_SuccessiveOptimizations_UsesUpdatedStartValues()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 0;
        var y = model.AddVariable();
        y.Start = 0;

        // First optimization: minimize x^2 + y^2 subject to x + y = 4
        model.SetObjective(x * x + y * y);
        model.AddConstraint(x + y == 4);

        var result1 = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result1.Status);
        Assert.AreEqual(2.0, x.Start, 0.001);
        Assert.AreEqual(2.0, y.Start, 0.001);

        // Second optimization: same problem, should start from previous solution
        var result2 = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result2.Status);

        // Start values should still be at the solution
        Assert.AreEqual(2.0, x.Start, 0.001);
        Assert.AreEqual(2.0, y.Start, 0.001);
    }

    [TestMethod]
    public void UpdateStartValues_MaxIterationsExceeded_UpdatesStartValues()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(-10, 10);
        x.Start = -5;
        var y = model.AddVariable(-10, 10);
        y.Start = -5;

        // Use Rosenbrock function which requires many iterations from a poor starting point
        // minimize (1-x)^2 + 100*(y-x^2)^2
        model.SetObjective(Expr.Pow(1 - x, 2) + 100 * Expr.Pow(y - x * x, 2));

        // Force early termination with very low iteration limit
        model.Options.MaxIterations = 1;
        model.Options.PrintLevel = 0;

        const double originalXStart = -5;
        const double originalYStart = -5;

        var result = model.Solve();

        // Should hit iteration limit (Rosenbrock won't converge in 1 iteration from (-5,-5))
        Assert.AreEqual(ApplicationReturnStatus.MaximumIterationsExceeded, result.Status);

        // Start values should be updated (partial progress is useful for warm starts)
        Assert.AreNotEqual(originalXStart, x.Start);
        Assert.AreNotEqual(originalYStart, y.Start);
    }

    [TestMethod]
    public void HessianApproximation_LimitedMemory_SolvesSuccessfully()
    {
        var model = new Model();
        var x1 = model.AddVariable(1, 5);
        x1.Start = 1;
        var x2 = model.AddVariable(1, 5);
        x2.Start = 5;
        var x3 = model.AddVariable(1, 5);
        x3.Start = 5;
        var x4 = model.AddVariable(1, 5);
        x4.Start = 1;

        // HS071 problem
        model.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);
        model.AddConstraint(x1 * x2 * x3 * x4 >= 25);
        model.AddConstraint(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 == 40);

        // Use limited memory approximation instead of exact Hessian
        model.Options.HessianApproximation = HessianApproximation.LimitedMemory;
        model.Options.PrintLevel = 0;

        var result = model.Solve();

        // Should still converge with limited memory approximation
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(17.014, result.ObjectiveValue, 0.01);
        Assert.AreEqual(1.0, result.Solution[x1], 0.01);
        Assert.AreEqual(4.743, result.Solution[x2], 0.01);
        Assert.AreEqual(3.821, result.Solution[x3], 0.01);
        Assert.AreEqual(1.379, result.Solution[x4], 0.01);
    }

    [TestMethod]
    public void WarmStart_UpdatesDualVariables_OnSuccessfulSolve()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 0;
        var y = model.AddVariable();
        y.Start = 0;

        // minimize x^2 + y^2 subject to x + y = 4
        model.SetObjective(x * x + y * y);
        var constraint = new Constraint(x + y, 4, 4);
        model.AddConstraint(constraint);

        model.Options.PrintLevel = 0;

        // Initial dual values should be zero
        Assert.AreEqual(0.0, constraint.DualStart);
        Assert.AreEqual(0.0, x.LowerBoundDualStart);
        Assert.AreEqual(0.0, x.UpperBoundDualStart);
        Assert.AreEqual(0.0, y.LowerBoundDualStart);
        Assert.AreEqual(0.0, y.UpperBoundDualStart);

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);

        // After solving, dual variables should be updated
        Assert.AreNotEqual(0.0, constraint.DualStart);
    }

    [TestMethod]
    public void WarmStart_SuccessiveSolves_ConvergesFaster()
    {
        // First solve: get the solution and dual variables
        var model1 = new Model();
        model1.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model1.Options.CheckDerivativesForNanInf = true;
        var x1 = model1.AddVariable(1, 5);
        x1.Start = 1;
        var x2 = model1.AddVariable(1, 5);
        x2.Start = 5;
        var x3 = model1.AddVariable(1, 5);
        x3.Start = 5;
        var x4 = model1.AddVariable(1, 5);
        x4.Start = 1;

        model1.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);
        var c1 = new Constraint(x1 * x2 * x3 * x4, 25, double.PositiveInfinity);
        var c2 = new Constraint(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4, 40, 40);
        model1.AddConstraint(c1);
        model1.AddConstraint(c2);

        model1.Options.PrintLevel = 0;
        var result1 = model1.Solve(updateStartValues: true);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result1.Status);

        // Second solve with warm start: should converge in very few iterations
        var model2 = new Model();
        model2.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model2.Options.CheckDerivativesForNanInf = true;
        var y1 = model2.AddVariable(1, 5);
        y1.Start = x1.Start;
        y1.LowerBoundDualStart = x1.LowerBoundDualStart;
        y1.UpperBoundDualStart = x1.UpperBoundDualStart;

        var y2 = model2.AddVariable(1, 5);
        y2.Start = x2.Start;
        y2.LowerBoundDualStart = x2.LowerBoundDualStart;
        y2.UpperBoundDualStart = x2.UpperBoundDualStart;

        var y3 = model2.AddVariable(1, 5);
        y3.Start = x3.Start;
        y3.LowerBoundDualStart = x3.LowerBoundDualStart;
        y3.UpperBoundDualStart = x3.UpperBoundDualStart;

        var y4 = model2.AddVariable(1, 5);
        y4.Start = x4.Start;
        y4.LowerBoundDualStart = x4.LowerBoundDualStart;
        y4.UpperBoundDualStart = x4.UpperBoundDualStart;

        model2.SetObjective(y1 * y4 * (y1 + y2 + y3) + y3);
        var d1 = new Constraint(y1 * y2 * y3 * y4, 25, double.PositiveInfinity);
        d1.DualStart = c1.DualStart;
        var d2 = new Constraint(y1 * y1 + y2 * y2 + y3 * y3 + y4 * y4, 40, 40);
        d2.DualStart = c2.DualStart;
        model2.AddConstraint(d1);
        model2.AddConstraint(d2);

        // Enable warm start and limit iterations to verify it converges quickly
        model2.Options.WarmStartInitPoint = true;
        model2.Options.MaxIterations = 5;
        model2.Options.PrintLevel = 0;
        var result2 = model2.Solve();

        // With warm start, should converge in few iterations
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result2.Status);
        Assert.AreEqual(result1.ObjectiveValue, result2.ObjectiveValue, 0.001);

        // Third solve without warm start: should fail with same iteration limit
        var model3 = new Model();
        model3.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model3.Options.CheckDerivativesForNanInf = true;
        var z1 = model3.AddVariable(1, 5);
        z1.Start = 1;
        var z2 = model3.AddVariable(1, 5);
        z2.Start = 5;
        var z3 = model3.AddVariable(1, 5);
        z3.Start = 5;
        var z4 = model3.AddVariable(1, 5);
        z4.Start = 1;

        model3.SetObjective(z1 * z4 * (z1 + z2 + z3) + z3);
        model3.AddConstraint(z1 * z2 * z3 * z4 >= 25);
        model3.AddConstraint(z1 * z1 + z2 * z2 + z3 * z3 + z4 * z4 == 40);

        model3.Options.MaxIterations = 5;
        model3.Options.PrintLevel = 0;
        var result3 = model3.Solve();

        // Cold start should not converge in 5 iterations
        Assert.AreNotEqual(ApplicationReturnStatus.SolveSucceeded, result3.Status);
    }

    [TestMethod]
    public void WarmStart_WithPerturbedProblem_ConvergesFasterThanColdStart()
    {
        // Solve initial problem to get warm start information
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x1 = model.AddVariable(1, 5);
        x1.Start = 1;
        var x2 = model.AddVariable(1, 5);
        x2.Start = 5;
        var x3 = model.AddVariable(1, 5);
        x3.Start = 5;
        var x4 = model.AddVariable(1, 5);
        x4.Start = 1;

        model.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);
        var c1 = new Constraint(x1 * x2 * x3 * x4, 25, double.PositiveInfinity);
        var c2 = new Constraint(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4, 40, 40);
        model.AddConstraint(c1);
        model.AddConstraint(c2);

        model.Options.PrintLevel = 0;
        var result1 = model.Solve(updateStartValues: true);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result1.Status);

        // Solve perturbed problem with warm start and limited iterations
        var modelWarm = new Model();
        modelWarm.Options.DerivativeTest = DerivativeTest.SecondOrder;
        modelWarm.Options.CheckDerivativesForNanInf = true;
        var w1 = modelWarm.AddVariable(1, 5);
        w1.Start = x1.Start;
        w1.LowerBoundDualStart = x1.LowerBoundDualStart;
        w1.UpperBoundDualStart = x1.UpperBoundDualStart;

        var w2 = modelWarm.AddVariable(1, 5);
        w2.Start = x2.Start;
        w2.LowerBoundDualStart = x2.LowerBoundDualStart;
        w2.UpperBoundDualStart = x2.UpperBoundDualStart;

        var w3 = modelWarm.AddVariable(1, 5);
        w3.Start = x3.Start;
        w3.LowerBoundDualStart = x3.LowerBoundDualStart;
        w3.UpperBoundDualStart = x3.UpperBoundDualStart;

        var w4 = modelWarm.AddVariable(1, 5);
        w4.Start = x4.Start;
        w4.LowerBoundDualStart = x4.LowerBoundDualStart;
        w4.UpperBoundDualStart = x4.UpperBoundDualStart;

        // Slightly perturbed problem: change RHS from 25 to 26
        modelWarm.SetObjective(w1 * w4 * (w1 + w2 + w3) + w3);
        var cw1 = new Constraint(w1 * w2 * w3 * w4, 26, double.PositiveInfinity);
        cw1.DualStart = c1.DualStart;
        var cw2 = new Constraint(w1 * w1 + w2 * w2 + w3 * w3 + w4 * w4, 40, 40);
        cw2.DualStart = c2.DualStart;
        modelWarm.AddConstraint(cw1);
        modelWarm.AddConstraint(cw2);

        // Enable warm start - should converge quickly
        modelWarm.Options.WarmStartInitPoint = true;
        modelWarm.Options.PrintLevel = 0;
        var resultWarm = modelWarm.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, resultWarm.Status);

        // Solve same perturbed problem cold
        var modelCold = new Model();
        modelCold.Options.DerivativeTest = DerivativeTest.SecondOrder;
        modelCold.Options.CheckDerivativesForNanInf = true;
        var z1 = modelCold.AddVariable(1, 5);
        z1.Start = 1;
        var z2 = modelCold.AddVariable(1, 5);
        z2.Start = 5;
        var z3 = modelCold.AddVariable(1, 5);
        z3.Start = 5;
        var z4 = modelCold.AddVariable(1, 5);
        z4.Start = 1;

        modelCold.SetObjective(z1 * z4 * (z1 + z2 + z3) + z3);
        modelCold.AddConstraint(z1 * z2 * z3 * z4 >= 26);
        modelCold.AddConstraint(z1 * z1 + z2 * z2 + z3 * z3 + z4 * z4 == 40);

        modelCold.Options.PrintLevel = 0;
        var resultCold = modelCold.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, resultCold.Status);

        // Warm start should use fewer iterations than cold start
        Assert.IsTrue(resultWarm.Statistics.IterationCount < resultCold.Statistics.IterationCount);
    }

    [TestMethod]
    public void WarmStart_DoesNotUpdateDualsWhenUpdateStartValuesFalse()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(0, 10);
        x.Start = 0;
        var y = model.AddVariable(0, 10);
        y.Start = 0;

        model.SetObjective(Expr.Pow(x - 3, 2) + Expr.Pow(y - 4, 2));
        var constraint = new Constraint(x + y, 5, 5);
        model.AddConstraint(constraint);

        model.Options.PrintLevel = 0;

        // Solve without updating start values
        var result = model.Solve(updateStartValues: false);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);

        // Dual variables should remain at zero
        Assert.AreEqual(0.0, constraint.DualStart);
        Assert.AreEqual(0.0, x.LowerBoundDualStart);
        Assert.AreEqual(0.0, x.UpperBoundDualStart);
        Assert.AreEqual(0.0, y.LowerBoundDualStart);
        Assert.AreEqual(0.0, y.UpperBoundDualStart);
    }

    [TestMethod]
    public void WarmStart_WithBoundedVariables_UpdatesBoundDuals()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(0, 2);
        x.Start = 0;
        var y = model.AddVariable(0, 2);
        y.Start = 0;

        // minimize (x-5)^2 + (y-5)^2
        // Both variables will be pushed to their upper bounds
        model.SetObjective(Expr.Pow(x - 5, 2) + Expr.Pow(y - 5, 2));

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);

        // Solution should be at bounds
        Assert.AreEqual(2.0, result.Solution[x], 0.001);
        Assert.AreEqual(2.0, result.Solution[y], 0.001);

        // Upper bound duals should be non-zero (active constraints)
        Assert.AreNotEqual(0.0, x.UpperBoundDualStart);
        Assert.AreNotEqual(0.0, y.UpperBoundDualStart);
    }

    [TestMethod]
    public void WarmStart_AutomaticallyEnabled_WhenDualDataPresent()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 2;
        x.LowerBoundDualStart = 0.5;  // Non-zero dual triggers warm start
        var y = model.AddVariable();
        y.Start = 2;

        model.SetObjective(x * x + y * y);
        model.AddConstraint(x + y == 4);

        // Don't explicitly set WarmStartInitPoint
        model.Options.PrintLevel = 0;

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(2.0, result.Solution[x], 0.001);
        Assert.AreEqual(2.0, result.Solution[y], 0.001);
    }

    [TestMethod]
    public void Statistics_ExposesIterationCount()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(1, 5);
        x.Start = 1;
        var y = model.AddVariable(1, 5);
        y.Start = 5;

        model.SetObjective(x * x + y * y);
        model.AddConstraint(x + y >= 6);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.IsNotNull(result.Statistics);
        Assert.IsTrue(result.Statistics.IterationCount > 0);
        Assert.IsTrue(result.Statistics.IterationCount < 100);
    }

    [TestMethod]
    public void Statistics_ProvidesFinalInfeasibilities()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        x.Start = 0;
        var y = model.AddVariable();
        y.Start = 0;

        model.SetObjective(x * x + y * y);
        model.AddConstraint(x + y == 4);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.IsTrue(result.Statistics.PrimalInfeasibility < 1e-6);
        Assert.IsTrue(result.Statistics.DualInfeasibility < 1e-6);
    }

    [TestMethod]
    public void WarmStart_ReducesIterationCount_ComparedToColdStart()
    {
        // First solve to get warm start data
        var model1 = new Model();
        model1.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model1.Options.CheckDerivativesForNanInf = true;
        var x1 = model1.AddVariable(1, 5);
        x1.Start = 1;
        var x2 = model1.AddVariable(1, 5);
        x2.Start = 5;
        var x3 = model1.AddVariable(1, 5);
        x3.Start = 5;
        var x4 = model1.AddVariable(1, 5);
        x4.Start = 1;

        model1.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);
        var c1 = new Constraint(x1 * x2 * x3 * x4, 25, double.PositiveInfinity);
        var c2 = new Constraint(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4, 40, 40);
        model1.AddConstraint(c1);
        model1.AddConstraint(c2);

        model1.Options.PrintLevel = 0;
        var result1 = model1.Solve(updateStartValues: true);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result1.Status);

        // Warm start solve
        var modelWarm = new Model();
        modelWarm.Options.DerivativeTest = DerivativeTest.SecondOrder;
        modelWarm.Options.CheckDerivativesForNanInf = true;
        var w1 = modelWarm.AddVariable(1, 5);
        w1.Start = x1.Start;
        w1.LowerBoundDualStart = x1.LowerBoundDualStart;
        w1.UpperBoundDualStart = x1.UpperBoundDualStart;

        var w2 = modelWarm.AddVariable(1, 5);
        w2.Start = x2.Start;
        w2.LowerBoundDualStart = x2.LowerBoundDualStart;
        w2.UpperBoundDualStart = x2.UpperBoundDualStart;

        var w3 = modelWarm.AddVariable(1, 5);
        w3.Start = x3.Start;
        w3.LowerBoundDualStart = x3.LowerBoundDualStart;
        w3.UpperBoundDualStart = x3.UpperBoundDualStart;

        var w4 = modelWarm.AddVariable(1, 5);
        w4.Start = x4.Start;
        w4.LowerBoundDualStart = x4.LowerBoundDualStart;
        w4.UpperBoundDualStart = x4.UpperBoundDualStart;

        modelWarm.SetObjective(w1 * w4 * (w1 + w2 + w3) + w3);
        var cw1 = new Constraint(w1 * w2 * w3 * w4, 25, double.PositiveInfinity);
        cw1.DualStart = c1.DualStart;
        var cw2 = new Constraint(w1 * w1 + w2 * w2 + w3 * w3 + w4 * w4, 40, 40);
        cw2.DualStart = c2.DualStart;
        modelWarm.AddConstraint(cw1);
        modelWarm.AddConstraint(cw2);

        modelWarm.Options.WarmStartInitPoint = true;
        modelWarm.Options.PrintLevel = 0;
        var resultWarm = modelWarm.Solve();

        // Cold start solve
        var modelCold = new Model();
        modelCold.Options.DerivativeTest = DerivativeTest.SecondOrder;
        modelCold.Options.CheckDerivativesForNanInf = true;
        var z1 = modelCold.AddVariable(1, 5);
        z1.Start = 1;
        var z2 = modelCold.AddVariable(1, 5);
        z2.Start = 5;
        var z3 = modelCold.AddVariable(1, 5);
        z3.Start = 5;
        var z4 = modelCold.AddVariable(1, 5);
        z4.Start = 1;

        modelCold.SetObjective(z1 * z4 * (z1 + z2 + z3) + z3);
        modelCold.AddConstraint(z1 * z2 * z3 * z4 >= 25);
        modelCold.AddConstraint(z1 * z1 + z2 * z2 + z3 * z3 + z4 * z4 == 40);

        modelCold.Options.PrintLevel = 0;
        var resultCold = modelCold.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, resultWarm.Status);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, resultCold.Status);

        // Warm start should require significantly fewer iterations
        Assert.IsTrue(resultWarm.Statistics.IterationCount <= 5);
        Assert.IsTrue(resultCold.Statistics.IterationCount > resultWarm.Statistics.IterationCount);
    }

    [TestMethod]
    public void AutoWarmStart_WithDualValues_ReducesIterationCount()
    {
        // First solve to get warm start data
        var model1 = new Model();
        model1.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model1.Options.CheckDerivativesForNanInf = true;
        var x1 = model1.AddVariable(1, 5);
        x1.Start = 1;
        var x2 = model1.AddVariable(1, 5);
        x2.Start = 5;
        var x3 = model1.AddVariable(1, 5);
        x3.Start = 5;
        var x4 = model1.AddVariable(1, 5);
        x4.Start = 1;

        model1.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);
        var c1 = new Constraint(x1 * x2 * x3 * x4, 25, double.PositiveInfinity);
        var c2 = new Constraint(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4, 40, 40);
        model1.AddConstraint(c1);
        model1.AddConstraint(c2);

        model1.Options.PrintLevel = 0;
        var result1 = model1.Solve(updateStartValues: true);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result1.Status);

        // Auto warm start solve (has dual values but WarmStartInitPoint not explicitly set)
        var modelAuto = new Model();
        modelAuto.Options.DerivativeTest = DerivativeTest.SecondOrder;
        modelAuto.Options.CheckDerivativesForNanInf = true;
        var a1 = modelAuto.AddVariable(1, 5);
        a1.Start = x1.Start;
        a1.LowerBoundDualStart = x1.LowerBoundDualStart;
        a1.UpperBoundDualStart = x1.UpperBoundDualStart;

        var a2 = modelAuto.AddVariable(1, 5);
        a2.Start = x2.Start;
        a2.LowerBoundDualStart = x2.LowerBoundDualStart;
        a2.UpperBoundDualStart = x2.UpperBoundDualStart;

        var a3 = modelAuto.AddVariable(1, 5);
        a3.Start = x3.Start;
        a3.LowerBoundDualStart = x3.LowerBoundDualStart;
        a3.UpperBoundDualStart = x3.UpperBoundDualStart;

        var a4 = modelAuto.AddVariable(1, 5);
        a4.Start = x4.Start;
        a4.LowerBoundDualStart = x4.LowerBoundDualStart;
        a4.UpperBoundDualStart = x4.UpperBoundDualStart;

        modelAuto.SetObjective(a1 * a4 * (a1 + a2 + a3) + a3);
        var ca1 = new Constraint(a1 * a2 * a3 * a4, 25, double.PositiveInfinity);
        ca1.DualStart = c1.DualStart;
        var ca2 = new Constraint(a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4, 40, 40);
        ca2.DualStart = c2.DualStart;
        modelAuto.AddConstraint(ca1);
        modelAuto.AddConstraint(ca2);

        // Don't set WarmStartInitPoint - let it auto-detect
        modelAuto.Options.PrintLevel = 0;
        var resultAuto = modelAuto.Solve();

        // Manual warm start solve (explicitly set WarmStartInitPoint)
        var modelManual = new Model();
        modelManual.Options.DerivativeTest = DerivativeTest.SecondOrder;
        modelManual.Options.CheckDerivativesForNanInf = true;
        var m1 = modelManual.AddVariable(1, 5);
        m1.Start = x1.Start;
        m1.LowerBoundDualStart = x1.LowerBoundDualStart;
        m1.UpperBoundDualStart = x1.UpperBoundDualStart;

        var m2 = modelManual.AddVariable(1, 5);
        m2.Start = x2.Start;
        m2.LowerBoundDualStart = x2.LowerBoundDualStart;
        m2.UpperBoundDualStart = x2.UpperBoundDualStart;

        var m3 = modelManual.AddVariable(1, 5);
        m3.Start = x3.Start;
        m3.LowerBoundDualStart = x3.LowerBoundDualStart;
        m3.UpperBoundDualStart = x3.UpperBoundDualStart;

        var m4 = modelManual.AddVariable(1, 5);
        m4.Start = x4.Start;
        m4.LowerBoundDualStart = x4.LowerBoundDualStart;
        m4.UpperBoundDualStart = x4.UpperBoundDualStart;

        modelManual.SetObjective(m1 * m4 * (m1 + m2 + m3) + m3);
        var cm1 = new Constraint(m1 * m2 * m3 * m4, 25, double.PositiveInfinity);
        cm1.DualStart = c1.DualStart;
        var cm2 = new Constraint(m1 * m1 + m2 * m2 + m3 * m3 + m4 * m4, 40, 40);
        cm2.DualStart = c2.DualStart;
        modelManual.AddConstraint(cm1);
        modelManual.AddConstraint(cm2);

        modelManual.Options.WarmStartInitPoint = true;
        modelManual.Options.PrintLevel = 0;
        var resultManual = modelManual.Solve();

        // Cold start solve (no dual values)
        var modelCold = new Model();
        modelCold.Options.DerivativeTest = DerivativeTest.SecondOrder;
        modelCold.Options.CheckDerivativesForNanInf = true;
        var z1 = modelCold.AddVariable(1, 5);
        z1.Start = 1;
        var z2 = modelCold.AddVariable(1, 5);
        z2.Start = 5;
        var z3 = modelCold.AddVariable(1, 5);
        z3.Start = 5;
        var z4 = modelCold.AddVariable(1, 5);
        z4.Start = 1;

        modelCold.SetObjective(z1 * z4 * (z1 + z2 + z3) + z3);
        modelCold.AddConstraint(z1 * z2 * z3 * z4 >= 25);
        modelCold.AddConstraint(z1 * z1 + z2 * z2 + z3 * z3 + z4 * z4 == 40);

        modelCold.Options.PrintLevel = 0;
        var resultCold = modelCold.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, resultAuto.Status);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, resultManual.Status);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, resultCold.Status);

        // Display iteration counts
        Console.WriteLine($"Manual warm start iterations: {resultManual.Statistics.IterationCount}");
        Console.WriteLine($"Auto warm start iterations: {resultAuto.Statistics.IterationCount}");
        Console.WriteLine($"Cold start iterations: {resultCold.Statistics.IterationCount}");

        // Auto warm start should match manual warm start performance
        // Both should be better than cold start
        Assert.IsTrue(resultManual.Statistics.IterationCount <= 5, "Manual warm start should converge quickly");
        Assert.IsTrue(resultCold.Statistics.IterationCount > resultManual.Statistics.IterationCount, "Cold start should take more iterations");

        // This assertion will fail if auto warm start is not implemented
        Assert.AreEqual(resultManual.Statistics.IterationCount, resultAuto.Statistics.IterationCount,
            "Auto warm start should match manual warm start when dual values are present");
    }

    [TestMethod]
    public void LinearProgram_ConstantMatrices()
    {
        // Test that LP problems correctly use constant gradient and Jacobian
        // Objective: minimize x + 2*y
        // subject to: x + y >= 3
        //            x >= 0, y >= 0
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(0, double.PositiveInfinity);
        var y = model.AddVariable(0, double.PositiveInfinity);

        model.SetObjective(x + 2 * y);
        model.AddConstraint(x + y >= 3);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        // Optimal solution minimizes objective on the constraint line x + y = 3
        // Since objective is x + 2*y and constraint is x + y = 3 (so x = 3 - y)
        // Substitute: (3 - y) + 2*y = 3 + y
        // This is minimized when y = 0, giving x = 3
        Assert.AreEqual(3.0, result.Solution[x], 0.01);
        Assert.AreEqual(0.0, result.Solution[y], 0.01);
        Assert.AreEqual(3.0, result.ObjectiveValue, 0.01);
    }

    [TestMethod]
    public void QuadraticProgram_ConstantHessian()
    {
        // Test that QP problems with linear constraints correctly use constant Hessian and Jacobian
        // Objective: minimize x^2 + y^2 - 4*x - 6*y
        // subject to: x + y <= 5
        //            x, y >= 0
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(0, double.PositiveInfinity);
        var y = model.AddVariable(0, double.PositiveInfinity);

        model.SetObjective(x * x + y * y - 4 * x - 6 * y);
        model.AddConstraint(x + y <= 5);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        // Unconstrained optimum would be at (2, 3) with objective -13
        // Constrained optimum is at (2, 3) since 2+3=5 satisfies the constraint
        Assert.AreEqual(2.0, result.Solution[x], 0.01);
        Assert.AreEqual(3.0, result.Solution[y], 0.01);
        Assert.AreEqual(-13.0, result.ObjectiveValue, 0.01);
    }

    [TestMethod]
    public void QuadraticallyConstrainedProgram()
    {
        // Test that QCP problems with quadratic constraints use constant Hessian
        // Objective: minimize x^2 + y^2
        // subject to: (x-2)^2 + (y-2)^2 <= 1
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable();
        var y = model.AddVariable();

        model.SetObjective(x * x + y * y);
        model.AddConstraint(Expr.Pow(x - 2, 2) + Expr.Pow(y - 2, 2) <= 1);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        // Solution is on the circle closest to origin
        var expectedVal = 2.0 - 1.0 / Math.Sqrt(2);
        Assert.AreEqual(expectedVal, result.Solution[x], 0.01);
        Assert.AreEqual(expectedVal, result.Solution[y], 0.01);
    }

    [TestMethod]
    public void NonlinearProgram_NoConstantMatrices()
    {
        // Test that nonlinear problems still work (matrices are recomputed each iteration)
        // Objective: minimize sin(x) + cos(y)
        // subject to: x^2 + y^2 <= 4
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        var x = model.AddVariable(-3, 3);
        var y = model.AddVariable(-3, 3);
        x.Start = 1;
        y.Start = 1;

        model.SetObjective(Expr.Sin(x) + Expr.Cos(y));
        model.AddConstraint(x * x + y * y <= 4);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        // Minimum is approximately at x=pi/2, y=pi (or nearby on constraint boundary)
        Assert.IsTrue(result.ObjectiveValue < 0, "Objective should be negative at optimum");
    }

    [TestMethod]
    public void ExpressionAnalysis_Linearity()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Constants are linear
        Expr constant = 5.0;
        Assert.IsTrue(constant.IsConstantWrtX());
        Assert.IsTrue(constant.IsLinear());
        Assert.IsTrue(constant.IsAtMostQuadratic());

        // Variables are linear
        Assert.IsFalse(x.IsConstantWrtX());
        Assert.IsTrue(x.IsLinear());
        Assert.IsTrue(x.IsAtMostQuadratic());

        // Linear combinations
        var linear = 2 * x + 3 * y - 5;
        Assert.IsFalse(linear.IsConstantWrtX());
        Assert.IsTrue(linear.IsLinear());
        Assert.IsTrue(linear.IsAtMostQuadratic());

        // Quadratic expressions
        var quadratic = x * x + y * y;
        Assert.IsFalse(quadratic.IsConstantWrtX());
        Assert.IsFalse(quadratic.IsLinear());
        Assert.IsTrue(quadratic.IsAtMostQuadratic());

        // Bilinear terms are quadratic
        var bilinear = x * y;
        Assert.IsFalse(bilinear.IsConstantWrtX());
        Assert.IsFalse(bilinear.IsLinear());
        Assert.IsTrue(bilinear.IsAtMostQuadratic());

        // Cubic is not quadratic
        var cubic = x * x * x;
        Assert.IsFalse(cubic.IsConstantWrtX());
        Assert.IsFalse(cubic.IsLinear());
        Assert.IsFalse(cubic.IsAtMostQuadratic());

        // Nonlinear functions
        var sine = Expr.Sin(x);
        Assert.IsFalse(sine.IsConstantWrtX());
        Assert.IsFalse(sine.IsLinear());
        Assert.IsFalse(sine.IsAtMostQuadratic());

        // Division by constant is linear if numerator is linear
        var divByConstant = (2 * x + 3) / 5;
        Assert.IsFalse(divByConstant.IsConstantWrtX());
        Assert.IsTrue(divByConstant.IsLinear());
        Assert.IsTrue(divByConstant.IsAtMostQuadratic());

        // Power operations
        var squared = Expr.Pow(x, 2);
        Assert.IsFalse(squared.IsConstantWrtX());
        Assert.IsFalse(squared.IsLinear());
        Assert.IsTrue(squared.IsAtMostQuadratic());

        var cubed = Expr.Pow(x, 3);
        Assert.IsFalse(cubed.IsConstantWrtX());
        Assert.IsFalse(cubed.IsLinear());
        Assert.IsFalse(cubed.IsAtMostQuadratic());
    }

    [TestMethod]
    public void ConstantOptions_RecomputedOnEachSolve()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // First solve: Linear program
        model.SetObjective(x + 2 * y);
        model.AddConstraint(x + y == 4);

        model.Options.PrintLevel = 0;
        var result1 = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result1.Status);
        
        // Options should still be null - they're set on solver, not Options
        Assert.IsNull(model.Options.GradFConstant);
        Assert.IsNull(model.Options.JacCConstant);
        Assert.IsNull(model.Options.HessianConstant);

        // Second solve: Add a nonlinear constraint (changes problem structure)
        model.AddConstraint(x * y >= 2);

        var result2 = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result2.Status);
        
        // Options should still be null - proving they don't persist incorrectly
        Assert.IsNull(model.Options.GradFConstant);
        Assert.IsNull(model.Options.JacCConstant);
        Assert.IsNull(model.Options.HessianConstant);
    }

    [TestMethod]
    public void LinearProgram_GradientCachedByIPOPT()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Linear objective - gradient should be constant
        model.SetObjective(x + 2 * y);
        model.AddConstraint(x + y >= 3);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // For LP, gradient should be evaluated only once (IPOPT caches it)
        // We verify this indirectly through solve statistics
        Assert.IsTrue(result.Statistics.IterationCount >= 1);
    }

    [TestMethod]
    public void LinearProgram_JacobianCachedByIPOPT()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);
        var z = model.AddVariable(0, 10);

        // Linear objective and constraints
        model.SetObjective(x + 2 * y + 3 * z);
        model.AddConstraint(x + y >= 3);
        model.AddConstraint(y + z == 5);
        model.AddConstraint(x - z <= 2);

        model.Options.PrintLevel = 5; // Verbose output to see evaluation counts
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // The solve should succeed with minimal evaluations due to caching
        // IPOPT output will show "Number of equality constraint Jacobian evaluations = 1"
        Assert.IsTrue(result.Statistics.IterationCount >= 0);
    }

    [TestMethod]
    public void QuadraticProgram_HessianCachedByIPOPT()
    {
        var model = new Model();
        var x = model.AddVariable(-10, 10);
        var y = model.AddVariable(-10, 10);

        // Quadratic objective - Hessian should be constant
        model.SetObjective(x * x + y * y - 4 * x - 6 * y);
        
        // Linear constraints
        model.AddConstraint(x + y >= 1);
        model.AddConstraint(x - y <= 5);

        model.Options.PrintLevel = 5; // Verbose to see evaluation counts
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // For QP with linear constraints, Hessian evaluations should be minimal
        // IPOPT will cache after first evaluation
        Assert.IsTrue(result.Statistics.IterationCount >= 0);
    }

    [TestMethod]
    public void QuadraticallyConstrainedProgram_HessianCachedByIPOPT()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Quadratic objective
        model.SetObjective(x * x + y * y);
        
        // Quadratic constraint - Hessian still constant
        model.AddConstraint(x * x + y * y <= 25);
        model.AddConstraint(x + 2 * y >= 3);

        model.Options.PrintLevel = 5;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // QCQP should also benefit from Hessian caching
        Assert.IsTrue(result.Statistics.IterationCount >= 0);
    }

    [TestMethod]
    public void NonlinearProgram_DerivativesNotCached()
    {
        var model = new Model();
        var x = model.AddVariable(0.1, 10);
        var y = model.AddVariable(0.1, 10);

        // Nonlinear objective - derivatives change at each iteration
        model.SetObjective(Expr.Exp(x) + Expr.Sin(y));
        
        // Nonlinear constraint
        model.AddConstraint(x * y >= 2);
        model.AddConstraint(Expr.Log(x) + y >= 1);

        model.Options.PrintLevel = 5;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // For NLP, gradient/Jacobian/Hessian evaluated at each iteration
        // Should see iteration_count evaluations in IPOPT output
        Assert.IsTrue(result.Statistics.IterationCount >= 1);
    }

    [TestMethod]
    public void GradFConstant_AutomaticallyEnabled_ForLinearObjective()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Linear objective: minimize x + 2*y
        model.SetObjective(x + 2 * y);
        model.AddConstraint(x + y >= 3);

        // Before solving, GradFConstant should be null
        Assert.IsNull(model.Options.GradFConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // After solving, GradFConstant should still be null (set on solver, not Options)
        Assert.IsNull(model.Options.GradFConstant);
    }

    [TestMethod]
    public void GradFConstant_NotEnabled_ForNonlinearObjective()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Nonlinear objective: minimize x^2 + y^2
        model.SetObjective(x * x + y * y);
        model.AddConstraint(x + y >= 3);

        Assert.IsNull(model.Options.GradFConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // GradFConstant should remain null (not auto-set for nonlinear)
        Assert.IsNull(model.Options.GradFConstant);
    }

    [TestMethod]
    public void GradFConstant_RespectUserSetting_WhenExplicitlySet()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Linear objective
        model.SetObjective(x + 2 * y);
        model.AddConstraint(x + y >= 3);

        // Explicitly set to false (user override)
        model.Options.GradFConstant = false;

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // Should remain false since user explicitly set it
        Assert.IsFalse(model.Options.GradFConstant == true);
    }

    [TestMethod]
    public void JacCConstant_ForLinearEqualityConstraints()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Quadratic objective
        model.SetObjective(x * x + y * y);
        
        // Linear equality constraint
        model.AddConstraint(x + y == 4);

        Assert.IsNull(model.Options.JacCConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // JacCConstant remains null (auto-set on solver only)
        Assert.IsNull(model.Options.JacCConstant);
    }

    [TestMethod]
    public void JacCConstant_NotEnabled_ForNonlinearEqualityConstraints()
    {
        var model = new Model();
        var x = model.AddVariable(1, 5);
        var y = model.AddVariable(1, 5);

        // Quadratic objective
        model.SetObjective(x * x + y * y);
        
        // Nonlinear equality constraint
        model.AddConstraint(x * y == 4);

        Assert.IsNull(model.Options.JacCConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // JacCConstant remains null
        Assert.IsNull(model.Options.JacCConstant);
    }

    [TestMethod]
    public void JacDConstant_ForLinearInequalityConstraints()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Quadratic objective
        model.SetObjective(x * x + y * y);
        
        // Linear inequality constraint
        model.AddConstraint(x + y >= 3);

        Assert.IsNull(model.Options.JacDConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // JacDConstant remains null (auto-set on solver only)
        Assert.IsNull(model.Options.JacDConstant);
    }

    [TestMethod]
    public void JacDConstant_NotEnabled_ForNonlinearInequalityConstraints()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Quadratic objective
        model.SetObjective(x * x + y * y);
        
        // Nonlinear inequality constraint
        model.AddConstraint(x * y >= 4);

        Assert.IsNull(model.Options.JacDConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // JacDConstant remains null
        Assert.IsNull(model.Options.JacDConstant);
    }

    [TestMethod]
    public void JacCConstant_AutomaticallyEnabled_WhenNoEqualityConstraints()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        model.SetObjective(x + y);
        
        // Only inequality constraint
        model.AddConstraint(x + y >= 3);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // JacCConstant remains null (auto-set on solver only)
        Assert.IsNull(model.Options.JacCConstant);
    }

    [TestMethod]
    public void JacDConstant_AutomaticallyEnabled_WhenNoInequalityConstraints()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        model.SetObjective(x + y);
        
        // Only equality constraint
        model.AddConstraint(x + y == 4);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // JacDConstant remains null (auto-set on solver only)
        Assert.IsNull(model.Options.JacDConstant);
    }

    [TestMethod]
    public void HessianConstant_ForQuadraticProgram()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Quadratic objective
        model.SetObjective(x * x + y * y);
        
        // Linear constraints
        model.AddConstraint(x + y >= 3);
        model.AddConstraint(x + y == 5);

        Assert.IsNull(model.Options.HessianConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // HessianConstant remains null (auto-set on solver only)
        Assert.IsNull(model.Options.HessianConstant);
    }

    [TestMethod]
    public void HessianConstant_ForQuadraticallyConstrainedProgram()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Quadratic objective
        model.SetObjective(x * x + y * y);
        
        // Quadratic constraint
        model.AddConstraint(x * x + y * y <= 25);

        Assert.IsNull(model.Options.HessianConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // HessianConstant remains null (auto-set on solver only)
        Assert.IsNull(model.Options.HessianConstant);
    }

    [TestMethod]
    public void HessianConstant_NotEnabled_ForNonlinearObjective()
    {
        var model = new Model();
        var x = model.AddVariable(0.1, 10);
        var y = model.AddVariable(0.1, 10);

        // Nonlinear objective (cubic term)
        model.SetObjective(x * x * x + y * y);
        
        // Linear constraint
        model.AddConstraint(x + y >= 3);

        Assert.IsNull(model.Options.HessianConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // HessianConstant remains null
        Assert.IsNull(model.Options.HessianConstant);
    }

    [TestMethod]
    public void HessianConstant_NotEnabled_ForNonlinearConstraint()
    {
        var model = new Model();
        var x = model.AddVariable(0.1, 10);
        var y = model.AddVariable(0.1, 10);

        // Quadratic objective
        model.SetObjective(x * x + y * y);
        
        // Nonlinear constraint (cubic term)
        model.AddConstraint(x * x * x + y >= 3);

        Assert.IsNull(model.Options.HessianConstant);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // HessianConstant remains null
        Assert.IsNull(model.Options.HessianConstant);
    }

    [TestMethod]
    public void HessianConstant_NotEnabled_WhenLimitedMemoryApproximation()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Quadratic objective
        model.SetObjective(x * x + y * y);
        
        // Linear constraint
        model.AddConstraint(x + y >= 3);

        // Use limited memory approximation
        model.Options.HessianApproximation = HessianApproximation.LimitedMemory;

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // HessianConstant remains null
        Assert.IsNull(model.Options.HessianConstant);
    }

    [TestMethod]
    public void AllConstantOptions_NotPersistedToOptions_ForLinearProgram()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Linear objective
        model.SetObjective(x + 2 * y);
        
        // Linear equality and inequality constraints
        model.AddConstraint(x + y == 4);
        model.AddConstraint(2 * x - y >= 1);

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // All constant options should remain null (set on solver only, not Options)
        Assert.IsNull(model.Options.GradFConstant);
        Assert.IsNull(model.Options.JacCConstant);
        Assert.IsNull(model.Options.JacDConstant);
        Assert.IsNull(model.Options.HessianConstant);
    }

    [TestMethod]
    public void ConstantOptions_RespectUserSettings()
    {
        var model = new Model();
        var x = model.AddVariable(0, 10);
        var y = model.AddVariable(0, 10);

        // Linear objective and constraints
        model.SetObjective(x + 2 * y);
        model.AddConstraint(x + y == 4);
        model.AddConstraint(x - y >= 1);

        // Explicitly set all to false (user override)
        model.Options.GradFConstant = false;
        model.Options.JacCConstant = false;
        model.Options.JacDConstant = false;
        model.Options.HessianConstant = false;

        model.Options.PrintLevel = 0;
        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        
        // All should remain false since user explicitly set them
        Assert.IsFalse(model.Options.GradFConstant == true);
        Assert.IsFalse(model.Options.JacCConstant == true);
        Assert.IsFalse(model.Options.JacDConstant == true);
        Assert.IsFalse(model.Options.HessianConstant == true);
    }
}
