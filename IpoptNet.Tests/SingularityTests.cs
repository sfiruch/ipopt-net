using Microsoft.VisualStudio.TestTools.UnitTesting;
using IpoptNet.Modelling;

namespace IpoptNet.Tests;

[TestClass]
public class SingularityTests
{
    [TestMethod]
    public void DivisionByZero_Recoverable()
    {
        var model = new Model();
        // x in [-2, 2]
        var x = model.AddVariable(-2, 2);
        x.Start = -0.5;

        // Minimize 1/x. 
        // For x < 0, this wants x to be as close to 0 as possible (from the left).
        // e.g. 1/(-0.00001) = -100000.
        // This will push x towards the singularity at 0.
        model.SetObjective(1.0 / x);

        var result = model.Solve();

        // It should NOT crash. 
        // It will likely hit the singularity and eventually stop because it can't 
        // evaluate closer to 0 or it reaches the maximum number of backtracks.
        System.Console.WriteLine($"Status: {result.Status}, x: {result.Solution[x]}, obj: {result.ObjectiveValue}");
        
        Assert.AreNotEqual(ApplicationReturnStatus.InternalError, result.Status);
    }

    [TestMethod]
    public void LogOfNegative_Recoverable()
    {
        var model = new Model();
        var x = model.AddVariable(-10, 10);
        x.Start = 1.0;

        // Minimize (x-2)^2 + log(x)
        // Optimum should be somewhere between 1 and 2.
        model.SetObjective(Expr.Pow(x - 2, 2) + Expr.Log(x));

        var result = model.Solve();

        System.Console.WriteLine($"Status: {result.Status}, x: {result.Solution[x]}, obj: {result.ObjectiveValue}");

        // It should find a solution where x > 0.
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.IsTrue(result.Solution[x] > 0);
    }
}
