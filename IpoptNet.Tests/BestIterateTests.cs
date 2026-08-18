using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class BestIterateTests
{
    /// <summary>On a clean solve the best iterate is the optimum, so it agrees with Solution.
    /// min x² + y² s.t. x + y == 4 has its optimum at (2, 2), objective 8.</summary>
    [TestMethod]
    public void CleanSolve_BestIterateMatchesSolution()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = -3;
        var y = model.AddVariable(-10, 10); y.Start = 9;
        model.AddConstraint(x + y == 4);
        model.SetObjective(Expr.Pow(x, 2) + Expr.Pow(y, 2));

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.IsNotNull(result.BestIterate);
        Assert.IsTrue(result.BestIterate.IsFeasible);
        Assert.AreEqual(2.0, result.BestIterate.Solution[x], 1e-6);
        Assert.AreEqual(2.0, result.BestIterate.Solution[y], 1e-6);
        Assert.AreEqual(result.ObjectiveValue, result.BestIterate.ObjectiveValue, 1e-6);
    }

    /// <summary>The case the feature exists for, and the reason "best" is feasibility-first rather
    /// than lowest-objective. Minimising x on the unit circle from an interior start, capped at 7
    /// iterations, IPOPT is mid-flight when it stops: its final iterate has objective -1.977 —
    /// below the true optimum of -1, because it sits far outside the circle with a constraint
    /// violation of ~5.8. The snapshot instead holds the iteration-3 point, objective 0.900 with a
    /// violation of ~0.004. A tracker that merely minimised the objective would have handed back
    /// the nonsense point.</summary>
    [TestMethod]
    public void TruncatedSolve_PrefersNearlyFeasibleOverLowerObjective()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        model.Options.MaxIterations = 7;
        var x = model.AddVariable(-2, 2); x.Start = 0.9;
        var y = model.AddVariable(-2, 2); y.Start = 0.1;
        model.AddConstraint(x * x + y * y == 1);
        model.SetObjective(x);

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.MaximumIterationsExceeded, result.Status);
        Assert.IsNotNull(result.BestIterate);

        Assert.IsTrue(result.BestIterate.PrimalInfeasibility < result.Statistics.PrimalInfeasibility,
            $"snapshot violation {result.BestIterate.PrimalInfeasibility:E3} must beat the returned "
            + $"iterate's {result.Statistics.PrimalInfeasibility:E3}");
        Assert.IsTrue(result.BestIterate.ObjectiveValue > result.ObjectiveValue,
            "The snapshot deliberately gives up objective to stay near-feasible; if this fails, the "
            + "policy has silently become lowest-objective-wins.");

        // The objective here is simply x, so the snapshot must be self-consistent.
        Assert.AreEqual(result.BestIterate.ObjectiveValue, result.BestIterate.Solution[x], 1e-9);
    }

    /// <summary>Conformance to the stated policy, across a sweep of truncation points. Note what
    /// the policy does NOT promise: the snapshot can be marginally LESS feasible than the returned
    /// iterate, because once two points are both inside constr_viol_tol they are equally "feasible"
    /// and the lower objective wins. On the converged run below the snapshot sits at ~1e-6 violation
    /// with an objective a hair under the true optimum, while Solve() returns the exactly-feasible
    /// point. That is tolerance-edge shopping, and it is the agreed behaviour — what the policy
    /// promises is only that a feasible point always beats an infeasible one.</summary>
    [TestMethod]
    public void Snapshot_ConformsToTheFeasibilityFirstPolicy()
    {
        const double tol = 1e-4;   // IPOPT's default constr_viol_tol, left unset here

        foreach (int cap in new[] { 3, 4, 6, 7, 8, 20 })
        {
            var model = new Model();
            model.Options.PrintLevel = 0;
            model.Options.MaxIterations = cap;
            var x = model.AddVariable(-2, 2); x.Start = 0.9;
            var y = model.AddVariable(-2, 2); y.Start = 0.1;
            model.AddConstraint(x * x + y * y == 1);
            model.SetObjective(x);

            var result = model.Solve();
            var best = result.BestIterate;
            Assert.IsNotNull(best, $"cap={cap}");

            Assert.AreEqual(best.PrimalInfeasibility <= tol, best.IsFeasible,
                $"cap={cap}: IsFeasible must agree with the violation and the tolerance.");

            bool finalIsFeasible = result.Statistics.PrimalInfeasibility <= tol;
            if (finalIsFeasible)
                Assert.IsTrue(best.ObjectiveValue <= result.ObjectiveValue + 1e-9,
                    $"cap={cap}: both feasible, so the snapshot must not have the worse objective "
                    + $"({best.ObjectiveValue:E6} vs {result.ObjectiveValue:E6}).");
            else
                Assert.IsTrue(best.PrimalInfeasibility <= result.Statistics.PrimalInfeasibility,
                    $"cap={cap}: the returned iterate is infeasible, so the snapshot must be at "
                    + $"least as close to feasible ({best.PrimalInfeasibility:E3} vs "
                    + $"{result.Statistics.PrimalInfeasibility:E3}).");
        }
    }

    /// <summary>The modelling layer scales variables itself — IPOPT is never told about
    /// Variable.Scale — so the raw iterate arrives in internal units and must be multiplied back
    /// out, exactly as the final solution readback does. With Scale left at 1 this bug is
    /// invisible, hence the deliberately non-unit scale here.</summary>
    [TestMethod]
    public void ScaledVariable_BestIterateIsInPhysicalUnits()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var big = model.AddVariable(0, 10_000, scale: 1000.0); big.Start = 100;
        model.SetObjective(Expr.Pow(big - 6000, 2));

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(6000.0, result.Solution![big], 1e-3);
        Assert.IsNotNull(result.BestIterate);
        Assert.AreEqual(result.Solution[big], result.BestIterate.Solution[big], 1e-6);
        Assert.AreEqual(6000.0, result.BestIterate.Solution[big], 1e-3);
    }

    /// <summary>When no iterate is ever feasible the snapshot still comes back, flagged infeasible
    /// and carrying the least-infeasible point rather than the lowest objective. x cannot satisfy
    /// both x >= 5 and x &lt;= 1.</summary>
    [TestMethod]
    public void InfeasibleProblem_ReportsLeastInfeasibleIterate()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        model.AddConstraint(x * 1 >= 5);
        model.AddConstraint(x * 1 <= 1);
        model.SetObjective(Expr.Pow(x, 2));

        var result = model.Solve();

        Assert.AreNotEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.IsNotNull(result.BestIterate);
        Assert.IsFalse(result.BestIterate.IsFeasible, "No iterate here can be feasible.");
        Assert.IsTrue(result.BestIterate.PrimalInfeasibility > 0);
    }

    /// <summary>Eliminated variables are not in IPOPT's decision vector, so the snapshot has to
    /// recompute them by running the block at the best x. v = 2p + 3, and the objective drives
    /// v to 9, so p = 3.</summary>
    [TestMethod]
    public void ImplicitBlock_SnapshotIncludesEliminatedVariables()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-10, 10); p.Start = 0;
        var v = model.AddVariable(); v.Start = 0;
        var c = model.AddConstraint(v - 2 * p - 3 == 0);
        model.AddImplicitBlock([v], [c]);
        model.SetObjective(Expr.Pow(v - 9, 2));

        var result = model.Solve();

        Assert.IsNotNull(result.BestIterate);
        Assert.AreEqual(3.0, result.BestIterate.Solution[p], 1e-6);
        Assert.AreEqual(9.0, result.BestIterate.Solution[v], 1e-6, "Eliminated variable must be reconstructed.");
        // The block relation must hold exactly at the snapshot, not merely approximately.
        Assert.AreEqual(2 * result.BestIterate.Solution[p] + 3, result.BestIterate.Solution[v], 1e-9);
    }

    /// <summary>Under partitioning the model-level snapshot is the partitions' snapshots taken
    /// together, with the objective constant added once — and it covers inert variables too, so a
    /// caller gets a complete vector without doing any partition bookkeeping of their own.</summary>
    [TestMethod]
    public void Partitioned_SnapshotCoversEveryVariable()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        var y = model.AddVariable(-10, 10); y.Start = 0;
        var inert = model.AddVariable(0, 10); inert.Start = 7;
        model.SetObjective(Expr.Pow(x - 3, 2) + Expr.Pow(y + 1, 2) + 100);

        var result = model.Solve();

        Assert.AreEqual(2, result.Partitions.Count);
        Assert.IsNotNull(result.BestIterate);
        Assert.AreEqual(3, result.BestIterate.Solution.Count, "Every variable, inert included.");
        Assert.AreEqual(3.0, result.BestIterate.Solution[x], 1e-6);
        Assert.AreEqual(-1.0, result.BestIterate.Solution[y], 1e-6);
        Assert.AreEqual(7.0, result.BestIterate.Solution[inert], 0.0);
        Assert.AreEqual(result.ObjectiveValue, result.BestIterate.ObjectiveValue, 1e-6);
        foreach (var partition in result.Partitions)
            Assert.IsNotNull(partition.BestIterate, "Each partition carries its own snapshot.");
    }

    /// <summary>The callback names the variables the current sub-problem is optimising, so a caller
    /// keeping their own records does not have to reverse-engineer the decomposition.</summary>
    [TestMethod]
    public void PartitionInfo_NamesTheCurrentPartitionsVariables()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        var y = model.AddVariable(-10, 10); y.Start = 0;
        model.SetObjective(Expr.Pow(x - 3, 2) + Expr.Pow(y + 1, 2));

        var seen = new Dictionary<int, IReadOnlyList<Variable>>();
        model.IntermediateCallback = (_, info) => { seen[info.Index] = info.Variables; return true; };
        model.Solve();

        Assert.AreEqual(2, seen.Count);
        CollectionAssert.AreEqual(new[] { x }, seen[0].ToArray());
        CollectionAssert.AreEqual(new[] { y }, seen[1].ToArray());
    }
}
