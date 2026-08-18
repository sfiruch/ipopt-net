using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class PartitioningTests
{
    /// <summary>Two components that share nothing: {x, y} joined by a constraint, and {a} on its
    /// own. Analytic optimum: x = 3, y = -1 (x + y = 2 satisfies x + y &lt;= 4 strictly, so the
    /// constraint is inactive and each square is minimised independently), and a = 7.</summary>
    private static (Model model, Variable x, Variable y, Variable a) BuildSeparable(bool partitioned)
    {
        var model = new Model { EnablePartitioning = partitioned };
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        var y = model.AddVariable(-10, 10); y.Start = 0;
        var a = model.AddVariable(0, 10); a.Start = 1;
        model.AddConstraint(x + y <= 4);
        model.SetObjective(Expr.Pow(x - 3, 2) + Expr.Pow(y + 1, 2) + Expr.Pow(a - 7, 2));
        return (model, x, y, a);
    }

    /// <summary>The headline guarantee: on a genuinely separable model, enabling partitioning
    /// changes nothing a caller can observe except that per-partition detail becomes available.</summary>
    [TestMethod]
    public void SeparableModel_FlagOnMatchesFlagOff()
    {
        var (joint, jx, jy, ja) = BuildSeparable(partitioned: false);
        var (split, sx, sy, sa) = BuildSeparable(partitioned: true);

        var jointResult = joint.Solve();
        var splitResult = split.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, jointResult.Status);
        Assert.AreEqual(jointResult.Status, splitResult.Status);
        Assert.AreEqual(jointResult.ObjectiveValue, splitResult.ObjectiveValue, 1e-7);
        Assert.AreEqual(jointResult.Solution![jx], splitResult.Solution![sx], 1e-7);
        Assert.AreEqual(jointResult.Solution[jy], splitResult.Solution[sy], 1e-7);
        Assert.AreEqual(jointResult.Solution[ja], splitResult.Solution[sa], 1e-7);

        // Sanity-check against the hand-derived optimum, so a shared bug can't make both agree.
        Assert.AreEqual(3.0, splitResult.Solution[sx], 1e-6);
        Assert.AreEqual(-1.0, splitResult.Solution[sy], 1e-6);
        Assert.AreEqual(7.0, splitResult.Solution[sa], 1e-6);

        Assert.AreEqual(2, splitResult.Partitions.Count);
        Assert.AreEqual(0, jointResult.Partitions.Count);
    }

    /// <summary>The objective's additive constant belongs to the model, not to any one partition.
    /// min (x-1)² + (y-2)² + 100 has optimum 100 — not 200, which is what counting the constant
    /// once per partition would give. Note the flattener expands the squares, so the completed-square
    /// constants join the 100 in the node's ConstantTerm: the objective is really
    /// 105 - 2x - 4y + x² + y², and at the optimum the two slices are worth -1 and -4.</summary>
    [TestMethod]
    public void SeparableModel_ObjectiveConstantCountedOnce()
    {
        var model = new Model { EnablePartitioning = true };
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        var y = model.AddVariable(-10, 10); y.Start = 0;
        model.SetObjective(Expr.Pow(x - 1, 2) + Expr.Pow(y - 2, 2) + 100);

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(2, result.Partitions.Count);
        Assert.AreEqual(100.0, result.ObjectiveValue, 1e-7);

        // Each slice carries only its own variable-bearing terms and no share of the constant.
        Assert.AreEqual(-1.0, result.Partitions[0].ObjectiveValue, 1e-7);
        Assert.AreEqual(-4.0, result.Partitions[1].ObjectiveValue, 1e-7);
        Assert.AreEqual(result.ObjectiveValue, result.Partitions.Sum(p => p.ObjectiveValue) + 105.0, 1e-7);
    }

    /// <summary>A model coupled both by a constraint and by a bilinear objective term must not
    /// decompose. It then takes the ordinary single-solve path, so the results are bit-identical
    /// to a flag-off run — no tolerance.</summary>
    [TestMethod]
    public void CoupledModel_YieldsSinglePartition()
    {
        static (Model model, Variable x, Variable y) Build(bool partitioned)
        {
            var model = new Model { EnablePartitioning = partitioned };
            model.Options.PrintLevel = 0;
            var x = model.AddVariable(-5, 5); x.Start = 0.3;
            var y = model.AddVariable(-5, 5); y.Start = 0.7;
            model.AddConstraint(x + y == 1);
            model.SetObjective(Expr.Pow(x, 2) + Expr.Pow(y, 2) + x * y);
            return (model, x, y);
        }

        var (split, sx, sy) = Build(partitioned: true);
        Assert.IsTrue(split.AnalyzePartitions().IsTrivial);
        Assert.AreEqual(1, split.AnalyzePartitions().Partitions.Count);

        var (joint, jx, jy) = Build(partitioned: false);
        var jointResult = joint.Solve();
        var splitResult = split.Solve();

        Assert.AreEqual(jointResult.Status, splitResult.Status);
        Assert.AreEqual(jointResult.ObjectiveValue, splitResult.ObjectiveValue);
        Assert.AreEqual(jointResult.Solution![jx], splitResult.Solution![sx]);
        Assert.AreEqual(jointResult.Solution[jy], splitResult.Solution[sy]);
        Assert.AreEqual(0, splitResult.Partitions.Count);
    }

    /// <summary>a·x + b·y is separable: a purely linear objective term contributes no coupling.
    /// Minimising 2x + 3y over x in [1,5], y in [2,6] drives both to their lower bounds.</summary>
    [TestMethod]
    public void LinearOnlyObjectiveCoupling_IsSeparable()
    {
        var model = new Model { EnablePartitioning = true };
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(1, 5); x.Start = 3;
        var y = model.AddVariable(2, 6); y.Start = 4;
        model.SetObjective(2 * x + 3 * y);

        Assert.AreEqual(2, model.AnalyzePartitions().Partitions.Count);

        var result = model.Solve();
        Assert.AreEqual(1.0, result.Solution![x], 1e-6);
        Assert.AreEqual(2.0, result.Solution[y], 1e-6);
        Assert.AreEqual(8.0, result.ObjectiveValue, 1e-6);
    }

    /// <summary>An implicit block is atomic: its eliminated variable and the parameters its
    /// residual reads must be solved together, even though the residual is not in _constraints.
    /// Here v is defined by v = 2p + 3 and the objective drives v to 9, so p = 3; w is a wholly
    /// independent component.</summary>
    [TestMethod]
    public void ImplicitBlock_IsAtomic()
    {
        var model = new Model { EnablePartitioning = true };
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-10, 10); p.Start = 0;
        var v = model.AddVariable(); v.Start = 0;
        var w = model.AddVariable(-10, 10); w.Start = 0;
        var c = model.AddConstraint(v - 2 * p - 3 == 0);
        model.AddImplicitBlock([v], [c]);
        model.SetObjective(Expr.Pow(v - 9, 2) + Expr.Pow(w - 5, 2));

        var partitioning = model.AnalyzePartitions();
        Assert.AreEqual(2, partitioning.Partitions.Count);

        var blockPartition = partitioning.Partitions.Single(x => x.ImplicitBlockCount == 1);
        CollectionAssert.Contains(blockPartition.Variables.ToArray(), p);
        CollectionAssert.Contains(blockPartition.EliminatedVariables.ToArray(), v);
        CollectionAssert.DoesNotContain(blockPartition.Variables.ToArray(), w);

        var result = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(3.0, result.Solution![p], 1e-6);
        Assert.AreEqual(9.0, result.Solution[v], 1e-6);
        Assert.AreEqual(5.0, result.Solution[w], 1e-6);
    }

    /// <summary>Chained blocks — the second reads the first's eliminated variable — form one
    /// component. Raw-mode collection is what makes that transitive: v2's residual reports v1
    /// itself rather than v1's inputs.</summary>
    [TestMethod]
    public void ChainedImplicitBlocks_StayInOnePartition()
    {
        var model = new Model { EnablePartitioning = true };
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-10, 10); p.Start = 0;
        var v1 = model.AddVariable(); v1.Start = 0;
        var v2 = model.AddVariable(); v2.Start = 0;
        var c1 = model.AddConstraint(v1 - 2 * p - 3 == 0);
        model.AddImplicitBlock([v1], [c1]);
        var c2 = model.AddConstraint(v2 - v1 - 1 == 0);
        model.AddImplicitBlock([v2], [c2]);
        model.SetObjective(Expr.Pow(v2 - 10, 2));

        var partitioning = model.AnalyzePartitions();
        Assert.AreEqual(1, partitioning.Partitions.Count);
        Assert.AreEqual(2, partitioning.Partitions[0].ImplicitBlockCount);

        // v2 = v1 + 1 = 2p + 4 = 10  =>  p = 3
        var result = model.Solve();
        Assert.AreEqual(3.0, result.Solution![p], 1e-6);
    }

    /// <summary>Variables the model never references — in no constraint, no implicit block and no
    /// objective term — never reach IPOPT at all. There is nothing to optimise and nothing to
    /// satisfy, so a solve would burn a full problem setup to hand back whatever the barrier drifted
    /// to. They are resolved from their start point instead: an explicit Start (clamped to bounds),
    /// otherwise the same bound-derived default IPOPT would have been seeded with. That makes the
    /// value deterministic and explainable, which the IPOPT round-trip never was.</summary>
    [TestMethod]
    public void InertVariables_ResolveToTheirStartPoint()
    {
        var model = new Model { EnablePartitioning = true };
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        var withStart = model.AddVariable(0, 10); withStart.Start = 3.5;
        var outOfBounds = model.AddVariable(0, 10); outOfBounds.Start = 42;   // clamped to the UB
        var noStart = model.AddVariable(-4, 8);                               // midpoint of the bounds
        var noStartNoUpper = model.AddVariable(2, double.PositiveInfinity);   // max(0, LB)
        model.SetObjective(Expr.Pow(x - 2, 2));

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(2.0, result.Solution![x], 1e-6);

        Assert.AreEqual(3.5, result.Solution[withStart], 0.0);
        Assert.AreEqual(10.0, result.Solution[outOfBounds], 0.0);
        Assert.AreEqual(2.0, result.Solution[noStart], 0.0);
        Assert.AreEqual(2.0, result.Solution[noStartNoUpper], 0.0);

        // The inert group is one partition in the analysis, but gets no IPOPT run of its own.
        var partitioning = model.AnalyzePartitions();
        Assert.AreEqual(2, partitioning.Partitions.Count);
        Assert.AreEqual(4, partitioning.Partitions.Single(p => p.IsInert).Variables.Count);
        Assert.AreEqual(1, result.Partitions.Count, "Only real sub-problems get solved.");
    }

    /// <summary>Not solving inert variables must not disturb the ones that matter: the referenced
    /// variables come back exactly as an unpartitioned solve leaves them.</summary>
    [TestMethod]
    public void InertVariables_DoNotAffectReferencedOnes()
    {
        static (Model model, Variable x, Variable inert) Build(bool partitioned)
        {
            var model = new Model { EnablePartitioning = partitioned };
            model.Options.PrintLevel = 0;
            var x = model.AddVariable(-10, 10); x.Start = 0;
            var inert = model.AddVariable(0, 10); inert.Start = 1;
            model.SetObjective(Expr.Pow(x - 2, 2));
            return (model, x, inert);
        }

        var (joint, jx, _) = Build(partitioned: false);
        var (split, sx, sInert) = Build(partitioned: true);

        var jointResult = joint.Solve();
        var splitResult = split.Solve();

        Assert.AreEqual(jointResult.Solution![jx], splitResult.Solution![sx], 1e-7);
        Assert.AreEqual(1.0, splitResult.Solution[sInert], 0.0);
    }

    /// <summary>A partition can own constraints but no objective terms — a pure feasibility
    /// sub-problem whose objective is the constant 0. With linear constraints that also means an
    /// empty Hessian structure (nele_hess == 0), which IPOPT must accept.</summary>
    [TestMethod]
    public void ConstraintOnlyPartition_ZeroObjective()
    {
        var model = new Model { EnablePartitioning = true };
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        var z = model.AddVariable(-10, 10); z.Start = 0;
        model.AddConstraint(z * 2 == 8);
        model.SetObjective(Expr.Pow(x - 1, 2));

        var partitioning = model.AnalyzePartitions();
        Assert.AreEqual(2, partitioning.Partitions.Count);
        var feasibilityPartition = partitioning.Partitions.Single(p => p.ObjectiveTermCount == 0);
        Assert.AreEqual(1, feasibilityPartition.Constraints.Count);

        var result = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(1.0, result.Solution![x], 1e-6);
        Assert.AreEqual(4.0, result.Solution[z], 1e-6);
    }

    /// <summary>Repeated analysis is reproducible: order never depends on hash-set iteration order.
    /// Both components here are the same size, so the tie-break on smallest Variable.Index decides,
    /// which is what the index assertion below pins. Analysis is also pure — it must not disturb a
    /// subsequent Solve.</summary>
    [TestMethod]
    public void AnalyzePartitions_IsDeterministicAndOrdered()
    {
        var model = new Model { EnablePartitioning = false };   // analysis is independent of the flag
        model.Options.PrintLevel = 0;
        // Declared interleaved so index order and component order are not trivially the same.
        var a1 = model.AddVariable(-5, 5); a1.Start = 0;
        var b1 = model.AddVariable(-5, 5); b1.Start = 0;
        var a2 = model.AddVariable(-5, 5); a2.Start = 0;
        var b2 = model.AddVariable(-5, 5); b2.Start = 0;
        model.AddConstraint(a1 + a2 == 2);
        model.AddConstraint(b1 + b2 == 6);
        model.SetObjective(Expr.Pow(a1, 2) + Expr.Pow(a2, 2) + Expr.Pow(b1, 2) + Expr.Pow(b2, 2));

        var first = model.AnalyzePartitions();
        var second = model.AnalyzePartitions();

        Assert.AreEqual(2, first.Partitions.Count);
        Assert.AreEqual(first.Partitions.Count, second.Partitions.Count);
        for (int p = 0; p < first.Partitions.Count; p++)
            CollectionAssert.AreEqual(first.Partitions[p].Variables.ToArray(),
                                      second.Partitions[p].Variables.ToArray());

        // Equal size (2 variables + 1 constraint each), so the Variable.Index tie-break orders them.
        for (int p = 1; p < first.Partitions.Count; p++)
        {
            Assert.AreEqual(
                first.Partitions[p - 1].Variables.Count + first.Partitions[p - 1].Constraints.Count,
                first.Partitions[p].Variables.Count + first.Partitions[p].Constraints.Count,
                "fixture assumption: the components are the same size");
            Assert.IsTrue(first.Partitions[p - 1].Variables[0].Index < first.Partitions[p].Variables[0].Index,
                "Equal-sized partitions must be ordered by ascending minimum Variable.Index.");
        }

        // a1 = a2 = 1, b1 = b2 = 3  =>  objective 2·1 + 2·9 = 20
        var result = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(20.0, result.ObjectiveValue, 1e-6);
    }

    [TestMethod]
    public void AnalyzePartitions_NoObjective_Throws()
    {
        var model = new Model();
        model.AddVariable(0, 1);
        Assert.ThrowsExactly<InvalidOperationException>(() => model.AnalyzePartitions());
    }

    /// <summary>A failing partition must not suppress the others: partitions are independent, so
    /// one failure says nothing about the rest, and callers rely on their per-iteration callback
    /// seeing every sub-problem. The aggregate status still reports the failure.</summary>
    [TestMethod]
    public void FailingPartition_DoesNotSuppressOthers()
    {
        var model = new Model { EnablePartitioning = true };
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        var u = model.AddVariable(-10, 10); u.Start = 0;
        model.AddConstraint(u * 1 >= 1);
        model.AddConstraint(u * 1 <= 0);   // together with the above: infeasible
        model.SetObjective(Expr.Pow(x - 2, 2));

        Assert.AreEqual(2, model.AnalyzePartitions().Partitions.Count);

        var result = model.Solve();

        Assert.AreEqual(2, result.Partitions.Count, "Every partition must be attempted.");
        Assert.AreNotEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded,
            result.Partitions.Single(p => p.Solution!.ContainsKey(x)).Status);
        // The converged partition still wrote its Start back, for warm-starting the next solve.
        Assert.AreEqual(2.0, x.Start!.Value, 1e-6);
    }

    /// <summary>The callback sees every partition, and the statistics it receives describe the
    /// whole model — cumulative iterations, and an objective that already accounts for the
    /// partitions that are done and the ones that have not started.</summary>
    [TestMethod]
    public void Callback_SeesEveryPartition_AndNormalisedObjective()
    {
        var (model, _, _, _) = BuildSeparable(partitioned: true);

        var seen = new List<(SolveStatistics stats, PartitionInfo info)>();
        model.IntermediateCallback = (stats, info) =>
        {
            seen.Add((stats, info));
            return true;
        };

        var result = model.Solve();

        Assert.IsTrue(seen.Count > 0);
        Assert.IsTrue(seen.All(s => s.info.Count == 2));
        CollectionAssert.AreEquivalent(new[] { 0, 1 }, seen.Select(s => s.info.Index).Distinct().ToArray());

        // Cumulative across partitions, so a progress bar built on it never runs backwards.
        for (int i = 1; i < seen.Count; i++)
            Assert.IsTrue(seen[i].stats.IterationCount >= seen[i - 1].stats.IterationCount,
                "Reported IterationCount must be cumulative across partitions.");

        // The very last callback is the last partition's final iterate, with every other partition
        // already at its optimum — so the reported objective is the model objective.
        Assert.AreEqual(result.ObjectiveValue, seen[^1].stats.ObjectiveValue, 1e-6);

        // The raw per-partition value is still available, and restarts per partition.
        Assert.AreEqual(0, seen.First(s => s.info.Index == 1).info.LocalStatistics.IterationCount);
    }

    /// <summary>Solving a partitioned model twice. This is the case the warm-start guard protects:
    /// the first solve writes back non-zero bound duals, which would auto-enable
    /// warm_start_init_point on the second — and IPOPT answers that with UnrecoverableException on
    /// a sub-problem with no constraints, which partitioning produces routinely.</summary>
    [TestMethod]
    public void RepeatedPartitionedSolve_WithConstraintFreePartition()
    {
        var model = new Model { EnablePartitioning = true };
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        var y = model.AddVariable(-10, 10); y.Start = 0;
        model.AddConstraint(x * 1 <= 8);          // partition {x} has a constraint
        model.SetObjective(Expr.Pow(x - 2, 2) + Expr.Pow(y - 5, 2));   // partition {y} has none

        Assert.AreEqual(2, model.AnalyzePartitions().Partitions.Count);

        var first = model.Solve();
        var second = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, first.Status, "first solve");
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, second.Status, "second solve");
        Assert.AreEqual(2.0, second.Solution![x], 1e-6);
        Assert.AreEqual(5.0, second.Solution[y], 1e-6);
    }

    /// <summary>Not partitioning-specific. ImplicitBlock._generation is never reset, so the
    /// evaluation-pass counter must be monotonic across Solve calls too — otherwise the second
    /// solve's first pass short-circuits every block and evaluates against stale eliminated
    /// values.</summary>
    [TestMethod]
    public void RepeatedSolve_ImplicitBlockGenerationRegression()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-10, 10); p.Start = 0;
        var v = model.AddVariable(); v.Start = 0;
        var c = model.AddConstraint(v - 2 * p - 3 == 0);
        model.AddImplicitBlock([v], [c]);
        model.SetObjective(Expr.Pow(v - 9, 2));

        var first = model.Solve();
        var second = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, first.Status);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, second.Status);
        Assert.AreEqual(3.0, first.Solution![p], 1e-6);
        Assert.AreEqual(3.0, second.Solution![p], 1e-6);
        Assert.AreEqual(9.0, second.Solution[v], 1e-6);
    }

    /// <summary>MaxIterations is a per-partition guard, deliberately not divided across partitions:
    /// it exists to stop one sub-problem spinning forever, and sharing it would make a later
    /// partition fail merely for having followed a hard one. So a 2-partition model can legitimately
    /// report more total iterations than the limit.</summary>
    [TestMethod]
    public void MaxIterations_AppliesPerPartition()
    {
        const int limit = 2;
        var (model, _, _, _) = BuildSeparable(partitioned: true);
        model.Options.MaxIterations = limit;

        var result = model.Solve();

        Assert.AreEqual(2, result.Partitions.Count);
        foreach (var partition in result.Partitions)
        {
            Assert.AreEqual(ApplicationReturnStatus.MaximumIterationsExceeded, partition.Status);
            Assert.AreEqual(limit, partition.Statistics.IterationCount);
        }
        Assert.AreEqual(2 * limit, result.Statistics.IterationCount,
            "Each partition gets its own iteration budget, so the total may exceed MaxIterations.");
    }

    /// <summary>MaxWallTime is a model-wide deadline, not a per-partition one: each partition is
    /// handed what remains of the budget, so N partitions cannot take N times as long as the caller
    /// allowed. Verified against IPOPT's own echo of the options in effect (print_user_options), so
    /// the assertion is on the budget actually handed to the solver rather than on elapsed
    /// wall-clock time, which would be flaky.</summary>
    [TestMethod]
    public void MaxWallTime_IsAModelWideDeadline()
    {
        const double budget = 100.0;
        var (model, _, _, _) = BuildSeparable(partitioned: true);
        model.Options.MaxWallTime = budget;
        model.Options.PrintUserOptions = true;
        model.Options.FilePrintLevel = 5;
        var logPath = Path.Combine(Path.GetTempPath(), $"ipopt-walltime-{Guid.NewGuid():N}.txt");
        model.Options.OutputFile = logPath;

        var result = model.Solve();
        var effective = File.ReadAllLines(logPath)
            .Where(l => l.Contains("max_wall_time"))
            .Select(l => double.Parse(l.Split('=')[1].Trim().Split(' ')[0],
                        System.Globalization.CultureInfo.InvariantCulture))
            .ToList();
        File.Delete(logPath);

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(2, effective.Count, "One budget echoed per partition.");
        Assert.IsTrue(effective.All(t => t <= budget), $"No partition may exceed the deadline: [{string.Join(", ", effective)}]");
        Assert.IsTrue(effective[1] < effective[0],
            $"The second partition must get the remainder, not the full budget: [{string.Join(", ", effective)}]");
    }

    /// <summary>Partitions are solved smallest first, so that under a model-wide time budget as many
    /// sub-problems as possible finish before the deadline, and BestIterate fills with the cheap wins
    /// rather than waiting behind a slow partition. The big component is declared FIRST here, so
    /// ordering by declaration (or by Variable.Index) would put it first — it must come second.</summary>
    [TestMethod]
    public void Partitions_AreOrderedSmallestFirst()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;

        // Big component, declared first: 4 variables + 3 constraints. The third constraint is what
        // makes it ONE component -- without it b1/b2 and b3/b4 would decompose further.
        var b1 = model.AddVariable(-10, 10); b1.Start = 0;
        var b2 = model.AddVariable(-10, 10); b2.Start = 0;
        var b3 = model.AddVariable(-10, 10); b3.Start = 0;
        var b4 = model.AddVariable(-10, 10); b4.Start = 0;
        model.AddConstraint(b1 + b2 == 2);
        model.AddConstraint(b3 + b4 == 6);
        model.AddConstraint(b1 + b3 == 1);
        // Medium component: 2 variables + 1 constraint.
        var m1 = model.AddVariable(-10, 10); m1.Start = 0;
        var m2 = model.AddVariable(-10, 10); m2.Start = 0;
        model.AddConstraint(m1 + m2 == 4);
        // Small component: 1 variable, no constraints.
        var s1 = model.AddVariable(-10, 10); s1.Start = 0;
        // Inert: referenced by nothing at all.
        var inert = model.AddVariable(0, 10); inert.Start = 5;

        model.SetObjective(
            Expr.Pow(b1, 2) + Expr.Pow(b2, 2) + Expr.Pow(b3, 2) + Expr.Pow(b4, 2)
            + Expr.Pow(m1, 2) + Expr.Pow(m2, 2)
            + Expr.Pow(s1 - 3, 2));

        var partitions = model.AnalyzePartitions().Partitions;

        Assert.AreEqual(4, partitions.Count);
        CollectionAssert.AreEqual(new[] { s1 }, partitions[0].Variables.ToArray(), "smallest first");
        CollectionAssert.AreEqual(new[] { m1, m2 }, partitions[1].Variables.ToArray());
        CollectionAssert.AreEqual(new[] { b1, b2, b3, b4 }, partitions[2].Variables.ToArray());
        Assert.IsTrue(partitions[3].IsInert, "the inert group sorts last");

        // Size is variables + constraints, and it must be non-decreasing across the solved ones.
        for (int i = 1; i < 3; i++)
            Assert.IsTrue(
                partitions[i - 1].Variables.Count + partitions[i - 1].Constraints.Count
                <= partitions[i].Variables.Count + partitions[i].Constraints.Count,
                $"partition {i} is smaller than partition {i - 1}");

        // Solved-partition indices agree across all three views, because the unsolved inert group
        // is last rather than somewhere in the middle.
        var seen = new Dictionary<int, IReadOnlyList<Variable>>();
        model.IntermediateCallback = (_, info) => { seen[info.Index] = info.Variables; return true; };
        var result = model.Solve();

        Assert.AreEqual(3, result.Partitions.Count, "the inert group is not solved");
        for (int i = 0; i < 3; i++)
            CollectionAssert.AreEqual(partitions[i].Variables.ToArray(), seen[i].ToArray(),
                $"PartitionInfo.Index {i} must mean the same partition AnalyzePartitions calls {i}");

        Assert.AreEqual(3.0, result.Solution![s1], 1e-6);
        Assert.AreEqual(5.0, result.Solution[inert], 0.0);
    }

    /// <summary>The core correctness property of the decomposition: what happens in one partition
    /// cannot reach another. A partition's expressions reference only its own variables — that is
    /// what the union-find guarantees — so the values other partitions left in the shared scratch
    /// buffer are never read, whether those are good values, degraded ones, or start values not yet
    /// touched. Partition A is solved first (it is smaller); its outcome is varied from clean
    /// convergence to a truncated, degraded stop, and B must be bit-identical every time.</summary>
    [TestMethod]
    public void OnePartitionsOutcomeCannotAffectAnother()
    {
        static (Model model, Variable a, Variable b1, Variable b2) Build(double aStart, int? cap)
        {
            var model = new Model();
            model.Options.PrintLevel = 0;
            if (cap is { } c) model.Options.MaxIterations = c;
            var a = model.AddVariable(-50, 50); a.Start = aStart;      // partition {a}, size 1
            var b1 = model.AddVariable(-10, 10); b1.Start = 0;         // partition {b1,b2}, size 3
            var b2 = model.AddVariable(-10, 10); b2.Start = 0;
            model.AddConstraint(b1 + b2 == 4);
            model.SetObjective(Expr.Pow(a - 7, 2) + Expr.Pow(b1, 2) + Expr.Pow(b2, 2));
            return (model, a, b1, b2);
        }

        var (clean, ca, cb1, cb2) = Build(aStart: 6.9, cap: null);
        var cleanResult = clean.Solve();

        var (far, _, fb1, fb2) = Build(aStart: -49, cap: null);
        var farResult = far.Solve();

        var (stopped, sa, sb1, sb2) = Build(aStart: -49, cap: 2);
        var stoppedResult = stopped.Solve();

        // Partition A really did end up somewhere different in the truncated run.
        Assert.AreEqual(ApplicationReturnStatus.MaximumIterationsExceeded, stoppedResult.Status);
        Assert.AreEqual(7.0, cleanResult.Solution![ca], 1e-6);
        Assert.IsTrue(Math.Abs(stoppedResult.Solution![sa] - 7.0) > 1e-3,
            "fixture assumption: the truncated run must leave A short of its optimum");

        // Partition B is unmoved by any of it — bit-identical, no tolerance.
        Assert.AreEqual(cleanResult.Solution[cb1], farResult.Solution![fb1]);
        Assert.AreEqual(cleanResult.Solution[cb2], farResult.Solution[fb2]);
        Assert.AreEqual(cleanResult.Solution[cb1], stoppedResult.Solution[sb1]);
        Assert.AreEqual(cleanResult.Solution[cb2], stoppedResult.Solution[sb2]);
    }

    /// <summary>A block can pin a variable with no decision-variable inputs at all (v == 5). Two
    /// things used to go wrong. The objective term referencing only v looked variable-free, because
    /// redirect-mode collection resolves an eliminated variable to its block's inputs and there are
    /// none — so it was stranded in an unrelated partition. And the resulting partition has no free
    /// variables, which is a zero-variable NLP that IPOPT refuses to create. Correct objective:
    /// (3-3)² + (5-9)² = 16.</summary>
    [TestMethod]
    public void BlockWithNoInputs_IsPartitionedAndResolvedCorrectly()
    {
        static (Model model, Variable x, Variable v) Build(bool partitioned)
        {
            var model = new Model { EnablePartitioning = partitioned };
            model.Options.PrintLevel = 0;
            var x = model.AddVariable(-10, 10); x.Start = 0;
            var v = model.AddVariable(); v.Start = 0;
            var c = model.AddConstraint(v - 5 == 0);
            model.AddImplicitBlock([v], [c]);
            model.SetObjective(Expr.Pow(x - 3, 2) + Expr.Pow(v - 9, 2));
            return (model, x, v);
        }

        var (split, sx, sv) = Build(partitioned: true);

        // Each partition owns exactly the objective term that belongs to it.
        var partitions = split.AnalyzePartitions().Partitions;
        Assert.AreEqual(2, partitions.Count);
        foreach (var partition in partitions)
            Assert.AreEqual(1, partition.ObjectiveTermCount,
                "the term referencing only the eliminated variable must land in ITS partition");

        var splitResult = split.Solve();
        var (joint, jx, jv) = Build(partitioned: false);
        var jointResult = joint.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, splitResult.Status);
        Assert.AreEqual(16.0, splitResult.ObjectiveValue, 1e-6);
        Assert.AreEqual(jointResult.ObjectiveValue, splitResult.ObjectiveValue, 1e-9);
        Assert.AreEqual(3.0, splitResult.Solution![sx], 1e-6);
        Assert.AreEqual(5.0, splitResult.Solution[sv], 1e-9);
        Assert.AreEqual(jointResult.Solution![jx], splitResult.Solution[sx], 1e-7);

        // Only the partition IPOPT actually solved reports a result.
        Assert.AreEqual(1, splitResult.Partitions.Count);
    }

    /// <summary>A constraint over only eliminated variables whose block has no inputs is a constant
    /// assertion — no decision can change it. IPOPT cannot be handed one: the row's Jacobian is empty,
    /// which the C API rejects when it is the only row ("Failed to create IPOPT problem") and trips a
    /// missing-key lookup in the Jacobian callback when it is not. Such constraints are now checked
    /// once and left out, so a satisfiable one simply solves — and the model is free to decompose
    /// around it, since it couples nothing.</summary>
    [TestMethod]
    public void ConstantConstraint_IsCheckedThenLeftOut()
    {
        static (Model model, Variable x, Variable v) Build(bool partitioned, double bound, bool alsoRealConstraint)
        {
            var model = new Model { EnablePartitioning = partitioned };
            model.Options.PrintLevel = 0;
            var x = model.AddVariable(-10, 10); x.Start = 0;
            var v = model.AddVariable(); v.Start = 0;
            var def = model.AddConstraint(v - 5 == 0);          // pins v, no decision inputs
            model.AddImplicitBlock([v], [def]);
            model.AddConstraint(v * 1 <= bound);                // references only v
            if (alsoRealConstraint) model.AddConstraint(x * 1 <= 8);
            model.SetObjective(Expr.Pow(x - 3, 2));
            return (model, x, v);
        }

        // Satisfiable, on its own: used to be "Failed to create IPOPT problem".
        foreach (bool partitioned in new[] { false, true })
        {
            var (model, x, v) = Build(partitioned, bound: 99, alsoRealConstraint: false);
            var result = model.Solve();
            Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status, $"partitioned={partitioned}");
            Assert.AreEqual(3.0, result.Solution![x], 1e-6);
            Assert.AreEqual(5.0, result.Solution[v], 1e-9);
        }

        // Satisfiable, beside a real constraint: used to throw a missing-key lookup.
        var (mixed, mx, _) = Build(partitioned: true, bound: 99, alsoRealConstraint: true);
        var mixedResult = mixed.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, mixedResult.Status);
        Assert.AreEqual(3.0, mixedResult.Solution![mx], 1e-6);

        // Coupling nothing, it no longer forces the model to stay in one piece.
        Assert.AreEqual(2, Build(true, 99, false).model.AnalyzePartitions().Partitions.Count);
    }

    /// <summary>A constant constraint that cannot hold is decided before the search begins, so it is
    /// reported as a modelling error naming the value and the bound it misses — rather than left for
    /// the caller to diagnose from a bare infeasible status.</summary>
    [TestMethod]
    public void ConstantConstraint_ThatCannotHold_IsRejected()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var x = model.AddVariable(-10, 10); x.Start = 0;
        var v = model.AddVariable(); v.Start = 0;
        var def = model.AddConstraint(v - 5 == 0);
        model.AddImplicitBlock([v], [def]);
        model.AddConstraint(v * 1 <= 4);          // v is pinned at 5; unsatisfiable
        model.SetObjective(Expr.Pow(x - 3, 2));

        var ex = Assert.ThrowsExactly<InvalidOperationException>(() => model.Solve());
        StringAssert.Contains(ex.Message, "references no decision variable");
        StringAssert.Contains(ex.Message, "5");
    }

    /// <summary>The neighbouring case must not regress: when the block DOES have decision inputs, the
    /// constraint is not constant at all — redirect-mode collection resolves v to those inputs, the
    /// Jacobian row is real, and IPOPT handles it normally. v = 2x + 3 with v &lt;= -99 forces x below
    /// its own lower bound, which is a genuine infeasibility for IPOPT to find.</summary>
    [TestMethod]
    public void ConstraintOnEliminatedVariableWithInputs_IsNotTreatedAsConstant()
    {
        static (Model model, Variable x) Build(double bound)
        {
            var model = new Model();
            model.Options.PrintLevel = 0;
            var x = model.AddVariable(-10, 10); x.Start = 0;
            var v = model.AddVariable(); v.Start = 0;
            var def = model.AddConstraint(v - 2 * x - 3 == 0);
            model.AddImplicitBlock([v], [def]);
            model.AddConstraint(v * 1 <= bound);
            model.SetObjective(Expr.Pow(x - 3, 2));
            return (model, x);
        }

        var (feasible, fx) = Build(99);
        var feasibleResult = feasible.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, feasibleResult.Status);
        Assert.AreEqual(3.0, feasibleResult.Solution![fx], 1e-6);

        var (infeasible, _) = Build(-99);
        Assert.AreEqual(ApplicationReturnStatus.InfeasibleProblemDetected, infeasible.Solve().Status,
            "a real Jacobian row means IPOPT reports infeasibility itself, not a modelling error");
    }

    /// <summary>Chained blocks in one partition alongside an independent component: the eliminated
    /// values must still be reconstructed through the chain, in registration (topological) order.
    /// v2 = v1 + 1 = 2p + 4, driven to 10, so p = 3.</summary>
    [TestMethod]
    public void ChainedBlocks_ResolveCorrectlyAlongsideAnotherPartition()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-10, 10); p.Start = 0;
        var v1 = model.AddVariable(); v1.Start = 0;
        var v2 = model.AddVariable(); v2.Start = 0;
        var c1 = model.AddConstraint(v1 - 2 * p - 3 == 0);
        model.AddImplicitBlock([v1], [c1]);
        var c2 = model.AddConstraint(v2 - v1 - 1 == 0);
        model.AddImplicitBlock([v2], [c2]);
        var w = model.AddVariable(-10, 10); w.Start = 0;
        model.SetObjective(Expr.Pow(v2 - 10, 2) + Expr.Pow(w - 5, 2));

        var partitions = model.AnalyzePartitions().Partitions;
        Assert.AreEqual(2, partitions.Count);
        Assert.AreEqual(2, partitions.Single(x => x.ImplicitBlockCount > 0).ImplicitBlockCount);

        var result = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(3.0, result.Solution![p], 1e-6);
        Assert.AreEqual(9.0, result.Solution[v1], 1e-6);
        Assert.AreEqual(10.0, result.Solution[v2], 1e-6);
        Assert.AreEqual(5.0, result.Solution[w], 1e-6);
        Assert.IsNotNull(result.BestIterate);
        Assert.AreEqual(10.0, result.BestIterate.Solution[v2], 1e-6);
    }
}
