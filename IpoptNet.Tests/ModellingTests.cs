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
}
