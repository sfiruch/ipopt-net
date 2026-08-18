using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class AutoEliminationTests
{
    /// <summary>The chain from ImplicitBlockTests, but with the blocks discovered rather than
    /// declared: implicit-Euler heat decay where each T[t+1] is defined by its step equation. The
    /// hand-built version eliminates every T[t+1]; automatic detection must find the same ones and
    /// reach the same optimum. Only the states are unbounded, so k and T[0] stay in the decision
    /// vector.</summary>
    [TestMethod]
    public void DiscoversTheSameEliminationsAsHandBuiltBlocks()
    {
        const double dt = 0.5, tOut = 5.0, trueT0 = 25.0, trueK = 0.4;
        const int steps = 6;
        var obs = new double[steps + 1];
        obs[0] = trueT0;
        for (int t = 0; t < steps; t++)
            obs[t + 1] = (obs[t] + dt * trueK * tOut) / (1 + dt * trueK);

        static (Model model, Variable k, Variable[] t) Build(bool auto)
        {
            var model = new Model { EnableAutomaticElimination = auto, EnablePartitioning = false };
            model.Options.PrintLevel = 0;
            var k = model.AddVariable(0.01, 5.0); k.Start = 1.0;
            var t = new Variable[steps + 1];
            t[0] = model.AddVariable(); t[0].Start = 20.0;
            for (int i = 0; i < steps; i++)
            {
                t[i + 1] = model.AddVariable();
                t[i + 1].Start = 20.0;
            }
            return (model, k, t);
        }

        void AddDynamics(Model model, Variable k, Variable[] t, bool declareBlocks)
        {
            for (int i = 0; i < steps; i++)
            {
                var c = model.AddConstraint(t[i + 1] + dt * k * t[i + 1] - t[i] - dt * k * tOut == 0);
                if (declareBlocks) model.AddImplicitBlock([t[i + 1]], [c]);
            }
            Expr obj = 0;
            for (int i = 0; i <= steps; i++)
                obj += Expr.Pow(t[i] - obs[i], 2);
            model.SetObjective(obj);
        }

        var (manual, mk, mt) = Build(auto: false);
        AddDynamics(manual, mk, mt, declareBlocks: true);
        var manualResult = manual.Solve();

        var (auto, ak, at) = Build(auto: true);
        AddDynamics(auto, ak, at, declareBlocks: false);

        // Detection finds exactly the states, one per step equation, and leaves k and T[0] alone.
        var candidates = auto.FindEliminableVariables();
        Assert.AreEqual(steps, candidates.Count);
        CollectionAssert.AreEquivalent(at.Skip(1).ToArray(), candidates.Select(c => c.Variable).ToArray());
        CollectionAssert.DoesNotContain(candidates.Select(c => c.Variable).ToArray(), ak);
        CollectionAssert.DoesNotContain(candidates.Select(c => c.Variable).ToArray(), at[0]);

        var autoResult = auto.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, manualResult.Status);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, autoResult.Status);
        Assert.AreEqual(trueK, autoResult.Solution![ak], 1e-4);
        Assert.AreEqual(manualResult.Solution![mk], autoResult.Solution[ak], 1e-6);
        for (int i = 0; i <= steps; i++)
            Assert.AreEqual(manualResult.Solution[mt[i]], autoResult.Solution[at[i]], 1e-6, $"T[{i}]");
    }

    /// <summary>Off by default, and either way the model comes back exactly as it was built. The
    /// flag is an option for the solve, not an edit: the restructuring lives only for the duration of
    /// the call. Both settings must reach the same answer.</summary>
    [TestMethod]
    public void OffByDefault_LeavesTheModelAlone()
    {
        static (Model model, Variable p, Variable v) Build(bool auto)
        {
            var model = new Model { EnableAutomaticElimination = auto };
            model.Options.PrintLevel = 0;
            var p = model.AddVariable(-100, 100); p.Start = 0;
            var v = model.AddVariable(); v.Start = 0;
            model.AddConstraint(v - 2 * p - 3 == 0);
            model.SetObjective(Expr.Pow(v - 9, 2));
            return (model, p, v);
        }

        var (plain, pp, pv) = Build(auto: false);
        Assert.IsFalse(new Model().EnableAutomaticElimination, "must default to off");
        var plainResult = plain.Solve();
        Assert.IsFalse(pv.IsEliminated, "nothing may be restructured with the flag off");

        var (eliminated, ep, ev) = Build(auto: true);
        var eliminatedResult = eliminated.Solve();

        // Structure restored, checked three ways. (Not by comparing ToString() before and after:
        // Solve writes back Start values, so the text legitimately differs.)
        Assert.IsFalse(ev.IsEliminated, "the restructuring must be undone before Solve returns");
        Assert.IsFalse(eliminated.ToString().Contains("Implicit blocks"), "no block may survive the call");
        Assert.AreEqual(1, eliminated.FindEliminableVariables().Count,
            "the equality is back in the constraint list, ready for a later solve");

        Assert.AreEqual(3.0, plainResult.Solution![pp], 1e-6);
        Assert.AreEqual(plainResult.Solution[pp], eliminatedResult.Solution![ep], 1e-6);
        Assert.AreEqual(plainResult.Solution[pv], eliminatedResult.Solution[ev], 1e-6);
    }

    /// <summary>Every eligibility rule, checked one at a time against a model where the only other
    /// candidate is known-good. A bounded variable is excluded because a block writes its value
    /// straight into the buffer; an inequality and a non-zero-RHS equality because AddImplicitBlock
    /// takes neither; a variable the constraint bends in because a block solves a linear system.</summary>
    [TestMethod]
    public void RejectsIneligibleCandidates()
    {
        static int CandidateCount(Action<Model, Variable> addConstraint, bool bounded = false)
        {
            var model = new Model();
            model.Options.PrintLevel = 0;
            var v = bounded ? model.AddVariable(-100, 100) : model.AddVariable();
            v.Start = 1;
            addConstraint(model, v);
            model.SetObjective(Expr.Pow(v - 9, 2));
            return model.FindEliminableVariables().Count;
        }

        Assert.AreEqual(1, CandidateCount((m, v) => m.AddConstraint(v - 3 == 0)),
            "baseline: unbounded, linear, equality at zero");
        Assert.AreEqual(0, CandidateCount((m, v) => m.AddConstraint(v - 3 == 0), bounded: true),
            "bounded variables cannot be eliminated");
        Assert.AreEqual(0, CandidateCount((m, v) => m.AddConstraint(v * 1 <= 3)),
            "inequalities are not definitions");
        Assert.AreEqual(0, CandidateCount((m, v) => m.AddConstraint(v * 1 == 3)),
            "AddImplicitBlock takes only expression == 0");
        Assert.AreEqual(0, CandidateCount((m, v) => m.AddConstraint(v * v - 4 == 0)),
            "the constraint must be linear in the variable it defines");
    }

    /// <summary>Two definitions that read each other cannot both be blocks — there is no order in
    /// which to register them. One is dropped and stays a decision variable; the other is still
    /// eliminated, and the answer is unchanged. a = b + 1 and b = a - 1 are the same line, so the
    /// objective picks the point on it nearest (4, 0).</summary>
    [TestMethod]
    public void BreaksCyclesRatherThanRefusing()
    {
        static (Model model, Variable a, Variable b) Build(bool auto)
        {
            var model = new Model { EnableAutomaticElimination = auto };
            model.Options.PrintLevel = 0;
            var a = model.AddVariable(); a.Start = 0;
            var b = model.AddVariable(); b.Start = 0;
            model.AddConstraint(a - b - 1 == 0);
            model.AddConstraint(b - a + 1 == 0);
            model.SetObjective(Expr.Pow(a - 4, 2) + Expr.Pow(b, 2));
            return (model, a, b);
        }

        var (model, a, b) = Build(auto: true);
        var candidates = model.FindEliminableVariables();
        Assert.AreEqual(1, candidates.Count, "exactly one of the mutually-defining pair survives");

        var result = model.Solve();
        var (plain, pa, pb) = Build(auto: false);
        var plainResult = plain.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(plainResult.Solution![pa], result.Solution![a], 1e-5);
        Assert.AreEqual(plainResult.Solution[pb], result.Solution![b], 1e-5);
        Assert.AreEqual(1.0, result.Solution[a] - result.Solution[b], 1e-6, "a = b + 1 must still hold");
    }

    /// <summary>Definitions are registered in dependency order: w is defined from v, so v's block has
    /// to come first or AddImplicitBlock rejects it. Declaration order here is deliberately the
    /// reverse. v = 2p + 3, w = v + 1, objective drives w to 10, so p = 3.</summary>
    [TestMethod]
    public void OrdersChainedDefinitionsByDependency()
    {
        var model = new Model { EnableAutomaticElimination = true };
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-100, 100); p.Start = 0;
        var v = model.AddVariable(); v.Start = 0;
        var w = model.AddVariable(); w.Start = 0;
        model.AddConstraint(w - v - 1 == 0);        // declared first, must be registered second
        model.AddConstraint(v - 2 * p - 3 == 0);
        model.SetObjective(Expr.Pow(w - 10, 2));

        var candidates = model.FindEliminableVariables();
        Assert.AreEqual(2, candidates.Count);
        Assert.AreSame(v, candidates[0].Variable, "the definition that reads nothing comes first");
        Assert.AreSame(w, candidates[1].Variable);

        var result = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(3.0, result.Solution![p], 1e-6);
        Assert.AreEqual(9.0, result.Solution[v], 1e-6);
        Assert.AreEqual(10.0, result.Solution[w], 1e-6);
    }

    /// <summary>Scaled variables are eligible now that blocks support them, and the discovered
    /// elimination reaches the same answer as leaving the variable in the decision vector.</summary>
    [TestMethod]
    public void ScaledVariablesAreEligible()
    {
        static (Model model, Variable p, Variable v) Build(bool auto)
        {
            var model = new Model { EnableAutomaticElimination = auto };
            model.Options.PrintLevel = 0;
            var p = model.AddVariable(-100, 100); p.Start = 0;
            var v = model.AddVariable(double.NegativeInfinity, double.PositiveInfinity, scale: 1000.0);
            v.Start = 0;
            model.AddConstraint(v - 2 * p - 3 == 0);
            model.SetObjective(Expr.Pow(v - 9, 2));
            return (model, p, v);
        }

        var (model, p, v) = Build(auto: true);
        Assert.AreEqual(1, model.FindEliminableVariables().Count, "a non-unit Scale is no longer a bar");

        var result = model.Solve();
        var (plain, pp, pv) = Build(auto: false);
        var plainResult = plain.Solve();

        Assert.AreEqual(3.0, result.Solution![p], 1e-6);
        Assert.AreEqual(9.0, result.Solution[v], 1e-6);
        Assert.AreEqual(plainResult.Solution![pp], result.Solution[p], 1e-6);
        Assert.AreEqual(plainResult.Solution[pv], result.Solution[v], 1e-6);
    }

    /// <summary>Analysis is pure: reporting candidates must not restructure anything, so a later
    /// solve with the flag off behaves exactly as if nothing had been inspected.</summary>
    [TestMethod]
    public void FindEliminableVariables_DoesNotMutate()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-100, 100); p.Start = 0;
        var v = model.AddVariable(); v.Start = 0;
        model.AddConstraint(v - 2 * p - 3 == 0);
        model.SetObjective(Expr.Pow(v - 9, 2));

        Assert.AreEqual(1, model.FindEliminableVariables().Count);
        Assert.AreEqual(1, model.FindEliminableVariables().Count, "repeatable");
        Assert.IsFalse(v.IsEliminated, "inspection must not restructure");

        var result = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(3.0, result.Solution![p], 1e-6);
        Assert.IsFalse(v.IsEliminated);
    }

    [TestMethod]
    public void FindEliminableVariables_NoObjective_Throws()
    {
        var model = new Model();
        model.AddVariable();
        Assert.ThrowsExactly<InvalidOperationException>(() => model.FindEliminableVariables());
    }

    /// <summary>The undo is complete enough to solve again, either way round, with the same answers.
    /// A restructuring that half-unwound would show up here as a second solve disagreeing with the
    /// first, or as the flag-off run inheriting blocks it never asked for.</summary>
    [TestMethod]
    public void RestructuringIsUndoneWellEnoughToSolveAgain()
    {
        var model = new Model { EnableAutomaticElimination = true };
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-100, 100); p.Start = 0;
        var v = model.AddVariable(); v.Start = 0;
        var w = model.AddVariable(); w.Start = 0;
        model.AddConstraint(w - v - 1 == 0);
        model.AddConstraint(v - 2 * p - 3 == 0);
        model.SetObjective(Expr.Pow(w - 10, 2));

        var first = model.Solve();

        // Flag off for the second run: it must see the equalities back as constraints.
        model.EnableAutomaticElimination = false;
        var second = model.Solve();

        // And on again for a third.
        model.EnableAutomaticElimination = true;
        var third = model.Solve();

        foreach (var (label, result) in new[] { ("first", first), ("second", second), ("third", third) })
        {
            Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status, label);
            Assert.AreEqual(3.0, result.Solution![p], 1e-6, label);
            Assert.AreEqual(9.0, result.Solution[v], 1e-6, label);
            Assert.AreEqual(10.0, result.Solution[w], 1e-6, label);
        }

        Assert.IsFalse(v.IsEliminated);
        Assert.IsFalse(w.IsEliminated);
    }
}
