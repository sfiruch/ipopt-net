using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class HessianStructureBugTests
{
    [TestMethod]
    public void ConstraintAddsHessianTermNotInObjective_ShouldNotThrow()
    {
        var model = new Model();
        
        // Variables
        var x1 = model.AddVariable(0, 10);
        x1.Start = 1;
        var x2 = model.AddVariable(0, 10);
        x2.Start = 1;
        
        // Objective only involves x1
        model.SetObjective(x1 * x1);
        
        // Constraint involves x1 * x2, which introduces Hessian term (x1, x2)
        // that doesn't appear in the objective
        model.AddConstraint(x1 * x2 == 5);
        
        // This should not throw KeyNotFoundException
        var result = model.Solve();
        
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
    }

    [TestMethod]
    public void QuadraticTermWithReversedIndices_ShouldNotThrow()
    {
        // Regression test for bug where QuadExpr.CollectHessianSparsityCore
        // was adding entries as (min, max) instead of (max, min),
        // causing KeyNotFoundException when HessianAccumulator normalized to (max, min)
        var model = new Model();
        
        var x1 = model.AddVariable(0, 10);
        x1.Start = 1;
        var x15 = model.AddVariable(0, 10);
        x15.Start = 1;
        
        // This creates a quadratic term between x1 (index 0) and x15 (index 1)
        // The bug was that QuadExpr would add sparsity entry as (0, 1) 
        // but HessianAccumulator would produce (1, 0)
        model.SetObjective(x1 * x15);
        
        // This should not throw KeyNotFoundException
        var result = model.Solve();
        
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
    }

    [TestMethod]
    public void DivisionWithNonConstantDenominator_ShouldNotThrow()
    {
        var model = new Model();
        
        // Many variables to increase likelihood of missing entries
        var vars = new Variable[20];
        for (int i = 0; i < 20; i++)
        {
            vars[i] = model.AddVariable(0.1, 10);
            vars[i].Start = 1;
        }
        
        // Objective: x[0] / x[1]  (division with variable denominator)
        model.SetObjective(vars[0] / vars[1]);
        
        // Constraint that involves other variables
        model.AddConstraint(vars[2] + vars[3] == 5);
        
        // This might expose the Hessian structure bug
        var result = model.Solve();
        
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
    }
}
