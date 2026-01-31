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
        var x2 = model.AddVariable(1, 5);
        var x3 = model.AddVariable(1, 5);
        var x4 = model.AddVariable(1, 5);

        // Objective: minimize x1*x4*(x1+x2+x3) + x3
        model.SetObjective(x1 * x4 * (x1 + x2 + x3) + x3);

        // Constraints
        model.AddConstraint((x1 * x2 * x3 * x4).GreaterThanOrEqual(25));
        model.AddConstraint((x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4).EqualTo(40));

        var result = model.Solve([1, 5, 5, 1]);

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
        var y = model.AddVariable();

        // f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        model.SetObjective(Expr.Pow(1 - x, 2) + 100 * Expr.Pow(y - x * x, 2));

        var result = model.Solve([-1, 1]);

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
        var y = model.AddVariable(0, 10);

        // minimize (x-3)^2 + (y-4)^2
        model.SetObjective(Expr.Pow(x - 3, 2) + Expr.Pow(y - 4, 2));

        var result = model.Solve([0, 0]);

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
        var y = model.AddVariable();

        // minimize x^2 + y^2
        // subject to x + y = 4
        model.SetObjective(x * x + y * y);
        model.AddConstraint((x + y).EqualTo(4));

        var result = model.Solve([0, 0]);

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

        // minimize -sin(x)
        // optimal at x = pi/2
        model.SetObjective(-Expr.Sin(x));

        var result = model.Solve([0]);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(Math.PI / 2, result.Solution[x], 0.01);
        Assert.AreEqual(-1.0, result.ObjectiveValue, 0.001);
    }

    [TestMethod]
    public void ExponentialExpression()
    {
        var model = new Model();
        var x = model.AddVariable(-5, 5);

        // minimize exp(x) - 2*x
        // derivative: exp(x) - 2 = 0 => x = ln(2)
        model.SetObjective(Expr.Exp(x) - 2 * x);

        var result = model.Solve([0]);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(Math.Log(2), result.Solution[x], 0.001);
    }

    [TestMethod]
    public void DivisionExpression()
    {
        var model = new Model();
        var x = model.AddVariable(0.1, 10);

        // minimize x + 1/x
        // derivative: 1 - 1/x^2 = 0 => x = 1
        model.SetObjective(x + 1 / x);

        var result = model.Solve([2]);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(1.0, result.Solution[x], 0.001);
        Assert.AreEqual(2.0, result.ObjectiveValue, 0.001);
    }
}
