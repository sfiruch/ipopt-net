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
        model.Options.LinearSolverOption = LinearSolver.Mumps;
        model.Options.HessianApproximationOption = HessianApproximation.Exact;
        model.Options.MuStrategyOption = MuStrategy.Adaptive;
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
        model.Options.LinearSolverOption = LinearSolver.Mumps;

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
        model.Options.LinearSolverOption = LinearSolver.PardisoMkl;

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
}
