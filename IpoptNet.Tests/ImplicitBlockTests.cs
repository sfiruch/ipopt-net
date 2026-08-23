using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class ImplicitBlockTests
{
    /// <summary>
    /// Single eliminated variable: v defined by v = 2*p + 3 (rewritten as v - 2*p - 3 == 0).
    /// Objective: (v - 7)^2  → optimum at v = 7  → p = 2.
    /// IPOPT's decision vector should contain only p; v is computed implicitly each pass.
    /// </summary>
    [TestMethod]
    public void SingleEliminatedVar_LinearDefinition()
    {
        var model = new Model();
        var p = model.AddVariable();
        p.Start = 0;

        var v = model.AddVariable();  // unbounded, scale=1 — required for elimination
        var defC = model.AddConstraint(v - 2 * p - 3 == 0);

        model.AddImplicitBlock(new[] { v }, new[] { defC });

        // Objective references v through VariableNode → redirect path → block solves.
        model.SetObjective(Expr.Pow(v - 7, 2));

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(2.0, result.Solution![p], 1e-4);
        Assert.AreEqual(7.0, result.Solution![v], 1e-4);
        Assert.AreEqual(0.0, result.ObjectiveValue, 1e-6);
    }

    /// <summary>
    /// Two-variable coupled implicit system at one timestep:
    ///   v1 = 0.5 * p + 0.4 * v2 + 1.0          (constraint:  v1 - 0.5*p - 0.4*v2 - 1 == 0)
    ///   v2 = 0.3 * p + 0.2 * v1 + 2.0          (constraint:  v2 - 0.3*p - 0.2*v1 - 2 == 0)
    /// Together: A·v = b where A = [[1, -0.4], [-0.2, 1]], b = [0.5p + 1, 0.3p + 2].
    /// det(A) = 1 - 0.08 = 0.92.
    ///
    /// Objective: (v1 - 5)^2 + (v2 - 4)^2.  Solve symbolically for the optimum p.
    /// v1(p) = (0.5p + 1 + 0.4*(0.3p + 2)) / 0.92 = (0.5p + 1 + 0.12p + 0.8) / 0.92
    ///       = (0.62p + 1.8) / 0.92
    /// v2(p) = (0.3p + 2 + 0.2*(0.5p + 1)) / 0.92 = (0.3p + 2 + 0.1p + 0.2) / 0.92
    ///       = (0.4p + 2.2) / 0.92
    ///
    /// d/dp of (v1 - 5)^2 + (v2 - 4)^2:
    ///   2*(v1 - 5)*(0.62/0.92) + 2*(v2 - 4)*(0.4/0.92) = 0
    ///   (v1 - 5)*0.62 + (v2 - 4)*0.4 = 0
    ///   ((0.62p + 1.8) - 5*0.92) * 0.62 + ((0.4p + 2.2) - 4*0.92) * 0.4 = 0       (multiplied by 0.92)
    ///   (0.62p + 1.8 - 4.6) * 0.62 + (0.4p + 2.2 - 3.68) * 0.4 = 0
    ///   (0.62p - 2.8) * 0.62 + (0.4p - 1.48) * 0.4 = 0
    ///   0.3844*p - 1.736 + 0.16*p - 0.592 = 0
    ///   0.5444*p = 2.328
    ///   p = 2.328 / 0.5444 ≈ 4.27627
    ///
    /// At that p:  v1 ≈ (0.62*4.27627 + 1.8)/0.92 ≈ 4.83836
    ///             v2 ≈ (0.4*4.27627 + 2.2)/0.92 ≈ 4.25055
    /// </summary>
    [TestMethod]
    public void CoupledTwoVarBlock_LinearLeastSquares()
    {
        var model = new Model();
        var p = model.AddVariable();
        p.Start = 0;

        var v1 = model.AddVariable();
        var v2 = model.AddVariable();

        var c1 = model.AddConstraint(v1 - 0.5 * p - 0.4 * v2 - 1.0 == 0);
        var c2 = model.AddConstraint(v2 - 0.3 * p - 0.2 * v1 - 2.0 == 0);

        model.AddImplicitBlock(new[] { v1, v2 }, new[] { c1, c2 });

        model.SetObjective(Expr.Pow(v1 - 5, 2) + Expr.Pow(v2 - 4, 2));

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(4.27627, result.Solution![p], 1e-3);
        Assert.AreEqual(4.83836, result.Solution![v1], 1e-3);
        Assert.AreEqual(4.25055, result.Solution![v2], 1e-3);
    }

    /// <summary>
    /// Cross-block dependency: implicit Euler of T'(t) = -k * (T - T_out) over 3 steps.
    /// Each step has its own ImplicitBlock; later blocks reference the previous block's eliminated v.
    /// Discretised: T[t+1] = T[t] + dt * (-k) * (T[t+1] - T_out)
    ///            ↔ T[t+1] * (1 + dt*k) - T[t] - dt*k*T_out = 0
    /// We have observations of T at the final step and fit k.
    /// </summary>
    [TestMethod]
    public void ChainedBlocks_ImplicitEuler_FitParameter()
    {
        var model = new Model();
        const double dt = 1.0;
        const double T_out = 10.0;
        const double T0 = 30.0;
        const double trueK = 0.5;
        const int nSteps = 3;

        // Generate observations at each step using the "true" k and exact implicit-Euler.
        var observed = new double[nSteps + 1];
        observed[0] = T0;
        for (int t = 0; t < nSteps; t++)
            observed[t + 1] = (observed[t] + dt * trueK * T_out) / (1 + dt * trueK);

        var k = model.AddVariable(0.01, 5.0);
        k.Start = 1.0;

        var T = new Variable[nSteps + 1];
        T[0] = model.AddVariable();             // initial state — also a decision var for now
        T[0].Start = T0;
        // Pin T[0] via an equality (instead of fixed bounds, which would prevent elimination):
        // we keep T[0] as a decision var here so the test exercises a non-elim "input" alongside k.
        model.AddConstraint(T[0] == T0);

        for (int t = 0; t < nSteps; t++)
        {
            T[t + 1] = model.AddVariable();   // unbounded, scale=1
            // (1 + dt*k) * T[t+1] - T[t] - dt*k*T_out == 0
            var residual = T[t + 1] + dt * k * T[t + 1] - T[t] - dt * k * T_out;
            var c = model.AddConstraint(residual == 0);
            model.AddImplicitBlock(new[] { T[t + 1] }, new[] { c });
        }

        // Objective: match T[nSteps] to observed[nSteps]
        model.SetObjective(Expr.Pow(T[nSteps] - observed[nSteps], 2));

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(trueK, result.Solution![k], 1e-3);
        for (int t = 0; t <= nSteps; t++)
            Assert.AreEqual(observed[t], result.Solution![T[t]], 1e-3, $"T[{t}] mismatch");
    }

    /// <summary>A non-unit Scale on an eliminated variable is supported. Nothing needs converting
    /// when the block writes v*: A and b are extracted in raw mode, where the variable's node already
    /// contributes its Scale, so the linear system is posed in evaluation-buffer units and v* comes
    /// out in them. What does need it are the redirect paths — the block's sensitivities are
    /// ∂scratch_v/∂scratch_input, and the value downstream expressions see is Scale·scratch_v.
    ///
    /// Here v is defined by v = 2p + 3 at Scale 1000, and the objective drives v to 9, so p = 3 —
    /// the same answer the unit-scale formulation gives. IPOPT's own second-order derivative checker
    /// validates the propagated gradient and Hessian rather than just the optimum.</summary>
    [TestMethod]
    public void Accepts_NonUnitScale()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-100, 100); p.Start = 0;
        var v = model.AddVariable(double.NegativeInfinity, double.PositiveInfinity, scale: 1000.0);
        v.Start = 0;
        var c = model.AddConstraint(v - 2 * p - 3 == 0);
        model.AddImplicitBlock([v], [c]);
        model.SetObjective(Expr.Pow(v - 9, 2));

        var result = ModellingTests.SolveWithDerivativeTest(model);

        ModellingTests.AssertDerivativeTestPassed(result.DerivativeTestResult);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(3.0, result.Solution![p], 1e-6);
        Assert.AreEqual(9.0, result.Solution[v], 1e-6, "the eliminated value is reported in physical units");
    }

    /// <summary>Scale on the second-order path. The test above cannot reach it: there v = 2p + 3 is
    /// linear in p, so ∂²v*/∂p² is identically zero and PropagateHessian contributes nothing whatever
    /// weight it is given. Here v·(1 + p) = 10 makes v* = 10/(1 + p), genuinely nonlinear in p, so the
    /// propagated Hessian is exercised. The residual is still linear in v — its coefficient is just
    /// (1 + p) rather than a constant — which is all an implicit block requires.
    ///
    /// The objective drives v to 2, so 10/(1 + p) = 2 and p = 4.</summary>
    [TestMethod]
    public void NonUnitScale_WithNonlinearSensitivity_HasCorrectHessian()
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(0, 10); p.Start = 1;
        var v = model.AddVariable(double.NegativeInfinity, double.PositiveInfinity, scale: 1000.0);
        v.Start = 0;
        var c = model.AddConstraint(v * (1 + p) - 10 == 0);
        model.AddImplicitBlock([v], [c]);
        model.SetObjective(Expr.Pow(v - 2, 2));

        var result = ModellingTests.SolveWithDerivativeTest(model);

        ModellingTests.AssertDerivativeTestPassed(result.DerivativeTestResult);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(4.0, result.Solution![p], 1e-6);
        Assert.AreEqual(2.0, result.Solution[v], 1e-6);
    }

    /// <summary>Scale must not change the answer. The same block solved at Scale 1 and Scale 1000
    /// has to agree, on the eliminated variable as well as the decision variable.</summary>
    [TestMethod]
    public void NonUnitScale_AgreesWithUnitScale()
    {
        static (Model model, Variable p, Variable v) Build(double scale)
        {
            var model = new Model();
            model.Options.PrintLevel = 0;
            var p = model.AddVariable(-100, 100); p.Start = 0;
            var v = model.AddVariable(double.NegativeInfinity, double.PositiveInfinity, scale: scale);
            v.Start = 0;
            var c = model.AddConstraint(v - 2 * p - 3 == 0);
            model.AddImplicitBlock([v], [c]);
            model.SetObjective(Expr.Pow(v - 9, 2) + Expr.Pow(p, 2) * 0.01);
            return (model, p, v);
        }

        var (plain, pp, pv) = Build(1.0);
        var (scaled, sp, sv) = Build(1000.0);
        var plainResult = plain.Solve();
        var scaledResult = scaled.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, plainResult.Status);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, scaledResult.Status);
        Assert.AreEqual(plainResult.Solution![pp], scaledResult.Solution![sp], 1e-6);
        Assert.AreEqual(plainResult.Solution[pv], scaledResult.Solution[sv], 1e-6);
        Assert.AreEqual(plainResult.ObjectiveValue, scaledResult.ObjectiveValue, 1e-6);
    }

    [TestMethod]
    public void Rejects_InequalityConstraint()
    {
        var model = new Model();
        var p = model.AddVariable();
        var v = model.AddVariable();
        var c = model.AddConstraint(v - p >= 0);
        Assert.ThrowsExactly<ArgumentException>(() => model.AddImplicitBlock(new[] { v }, new[] { c }));
    }

    [TestMethod]
    public void Rejects_ConstraintNotInModel()
    {
        var model = new Model();
        var p = model.AddVariable();
        var v = model.AddVariable();
        var orphan = new Constraint(v - p, 0, 0);
        Assert.ThrowsExactly<ArgumentException>(() => model.AddImplicitBlock(new[] { v }, new[] { orphan }));
    }

    /// <summary>Eliminating a variable that's already referenced by an earlier block's residual
    /// must fail — that's the within-model topological-order violation: the earlier block would
    /// solve first (registration order) and read v before this block had a chance to define it.</summary>
    [TestMethod]
    public void Rejects_VariableAlreadyUsedByEarlierBlock()
    {
        var model = new Model();
        var p = model.AddVariable();
        var v1 = model.AddVariable();
        var v2 = model.AddVariable();
        // Add B1 first, with a residual that references v2.
        var c1 = model.AddConstraint(v1 - v2 - p == 0);
        model.AddImplicitBlock(new[] { v1 }, new[] { c1 });
        // Now try to eliminate v2 — its earlier appearance in B1's residual makes this an out-of-order add.
        var c2 = model.AddConstraint(v2 - p == 0);
        var ex = Assert.ThrowsExactly<ArgumentException>(() =>
            model.AddImplicitBlock(new[] { v2 }, new[] { c2 }));
        StringAssert.Contains(ex.Message, "earlier implicit block");
    }

    /// <summary>Smoke test: registering blocks in valid topological order doesn't trigger the
    /// out-of-order check.</summary>
    [TestMethod]
    public void Accepts_TopologicalOrder()
    {
        var model = new Model();
        var p = model.AddVariable();
        var v1 = model.AddVariable();
        var v2 = model.AddVariable();
        var c1 = model.AddConstraint(v1 - p == 0);
        var c2 = model.AddConstraint(v2 - v1 - 1 == 0);  // v2 depends on v1 — v1's block must come first
        model.AddImplicitBlock(new[] { v1 }, new[] { c1 });
        model.AddImplicitBlock(new[] { v2 }, new[] { c2 });
        model.SetObjective(Expr.Pow(v2 - 5, 2));
        var result = model.Solve();
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(4.0, result.Solution![p], 1e-4);    // v2 = p + 1, want v2 = 5 → p = 4
        Assert.AreEqual(4.0, result.Solution![v1], 1e-4);
        Assert.AreEqual(5.0, result.Solution![v2], 1e-4);
    }

    // ----- Chained blocks' second-order sensitivities -----

    /// <summary>The Hessian propagated through a CHAIN of blocks, checked by IPOPT's own finite
    /// differences. The objective reads only the last state, so every earlier block is reached for
    /// the first time from a later one — the order in which the blocks' shared working buffers used
    /// to collide, leaving each block computing with its predecessor's local Hessians and returning
    /// derivatives that were wrong but plausible. Nothing failed at the time: the wrong numbers were
    /// still finite, still symmetric, and IPOPT simply converged more slowly on them.
    ///
    /// v_t·(1 + p) = v_{t−1} makes each step genuinely nonlinear in p, so the ν chain (a block's
    /// second-order sensitivity feeding the next one's) actually carries something. Starting from
    /// v_0 = 8 and driving the last state to 1 gives (1 + p)^nSteps = 8, i.e. p = 2^(3/nSteps) − 1.</summary>
    [TestMethod]
    [DataRow(2)]
    [DataRow(3)]
    [DataRow(5)]
    public void ChainedBlocks_HaveCorrectSecondOrderSensitivity(int nSteps)
    {
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(0, 10);
        p.Start = 0.5;

        Expr previous = 8.0;
        Variable? last = null;
        for (int t = 0; t < nSteps; t++)
        {
            var v = model.AddVariable();
            model.AddImplicitBlock([v], [model.AddConstraint(v * (1 + p) - previous == 0)]);
            previous = v;
            last = v;
        }
        // Only the final state is observed, so the chain is walked back-to-front on first evaluation.
        model.SetObjective(Expr.Pow(last! - 1.0, 2));

        var result = ModellingTests.SolveWithDerivativeTest(model);

        ModellingTests.AssertDerivativeTestPassed(result.DerivativeTestResult);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(Math.Pow(2.0, 3.0 / nSteps) - 1.0, result.Solution![p], 1e-6);
        Assert.AreEqual(1.0, result.Solution[last!], 1e-6);
    }

    /// <summary>The same chain with the states observed at every step, which is the order that always
    /// worked: each block is computed before the one that reads it, so the collision never arose.
    /// Kept alongside the test above so a regression that only restores the old ordering dependence
    /// is still visible as one passing and one failing.</summary>
    [TestMethod]
    public void ChainedBlocks_AllStatesObserved_HaveCorrectSecondOrderSensitivity()
    {
        const int nSteps = 4;
        var model = new Model();
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(0, 10);
        p.Start = 0.5;

        Expr previous = 8.0;
        var states = new List<Variable>();
        for (int t = 0; t < nSteps; t++)
        {
            var v = model.AddVariable();
            model.AddImplicitBlock([v], [model.AddConstraint(v * (1 + p) - previous == 0)]);
            previous = v;
            states.Add(v);
        }
        Expr objective = 0;
        foreach (var v in states)
            objective += Expr.Pow(v - 1.0, 2);
        model.SetObjective(objective);

        var result = ModellingTests.SolveWithDerivativeTest(model);

        ModellingTests.AssertDerivativeTestPassed(result.DerivativeTestResult);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
    }

    // ----- Inputs pinned by equal bounds are treated as constants -----

    /// <summary>Builds the two-block chain used by the pinned-input tests. Both blocks are nonlinear
    /// in the free parameter, so the second-order sensitivity path (and, on the second block, the ν
    /// chain through the first) is genuinely exercised rather than vanishing:
    ///   v1·(1 + p + q) = 10        →  v1 = 10 / (1 + p + q)
    ///   v2·(1 + p)     = v1        →  v2 = v1 / (1 + p)
    /// With q = 1 the objective (v2 − 1)² gives (2 + p)(1 + p) = 10, i.e. p = (−3 + √41)/2.
    /// <paramref name="pinQ"/> chooses whether q is pinned by equal bounds or free.</summary>
    private static (Model model, Variable p, Variable q, Variable v2) BuildPinnedChain(bool pinQ, double qValue = 1.0)
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(0, 10);
        p.Start = 1;
        var q = pinQ ? model.AddVariable(qValue, qValue) : model.AddVariable(0, 10);
        q.Start = qValue;

        var v1 = model.AddVariable();
        var v2 = model.AddVariable();
        var c1 = model.AddConstraint(v1 * (1 + p + q) - 10 == 0);
        model.AddImplicitBlock([v1], [c1]);
        var c2 = model.AddConstraint(v2 * (1 + p) - v1 == 0);
        model.AddImplicitBlock([v2], [c2]);

        model.SetObjective(Expr.Pow(v2 - 1, 2));
        return (model, p, q, v2);
    }

    private static readonly double ExpectedChainP = (-3 + Math.Sqrt(41)) / 2;

    /// <summary>Builds a single-block counterpart of <see cref="BuildPinnedChain"/>:
    ///   v·(1 + p + q) = 10  →  v = 10 / (1 + p + q)
    /// nonlinear in p, so the second-order path is exercised, and with q = 1 the objective (v − 2)²
    /// puts the optimum at 10/(2 + p) = 2, i.e. p = 3. <paramref name="qValue"/> null makes q a free
    /// variable instead of a pinned one.</summary>
    private static (Model model, Variable p, Variable? q, Variable v) BuildPinnedSingle(double? qValue)
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(0, 10);
        p.Start = 1;
        var q = qValue is { } pinned ? model.AddVariable(pinned, pinned) : model.AddVariable(0, 10);
        q.Start = qValue ?? 1.0;
        var v = model.AddVariable();
        model.AddImplicitBlock([v], [model.AddConstraint(v * (1 + p + q) - 10 == 0)]);
        model.SetObjective(Expr.Pow(v - 2, 2));
        return (model, p, q, v);
    }

    /// <summary>The soundness test, and the reason the optimization is allowed at all: a pinned input
    /// and the same value folded into the residual as a literal constant must give IPOPT the same
    /// problem. Not merely the same optimum — the same iterate, reached in the same number of
    /// iterations, bit for bit. That is only true because IPOPT removes fixed variables from the
    /// problem under its default fixed_variable_treatment, so the sensitivity and Hessian entries
    /// this change stops supplying for them were never consumed. If a future IPOPT stopped doing
    /// that, the trajectories would diverge here.</summary>
    [TestMethod]
    public void PinnedInput_MatchesConstantFoldedModel()
    {
        var (pinnedModel, pinnedP, q, pinnedV) = BuildPinnedSingle(qValue: 1.0);
        var pinned = pinnedModel.Solve();

        var model = new Model();
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(0, 10);
        p.Start = 1;
        var v = model.AddVariable();
        model.AddImplicitBlock([v], [model.AddConstraint(v * (1 + p + 1.0) - 10 == 0)]);
        model.SetObjective(Expr.Pow(v - 2, 2));
        var constant = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, pinned.Status);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, constant.Status);
        Assert.AreEqual(3.0, constant.Solution![p], 1e-6, "sanity: the constant-folded model finds p = 3");
        Assert.AreEqual(constant.Solution[p], pinned.Solution![pinnedP],
            "pinned and constant-folded must reach the same iterate exactly");
        Assert.AreEqual(constant.Solution[v], pinned.Solution[pinnedV]);
        Assert.AreEqual(constant.Statistics.IterationCount, pinned.Statistics.IterationCount,
            "same iteration count ⇒ IPOPT never consumed the dropped derivatives");
        Assert.AreEqual(1.0, pinned.Solution[q!], 1e-9, "the pinned input is still reported at its value");
    }

    /// <summary>IPOPT's own second-order derivative checker, run on the pinned model, passes — which
    /// it only can because the optimization stands down while the checker is on. Unlike the solve,
    /// the checker does ask about a fixed variable's entries, so it would report the ones this change
    /// otherwise omits as errors (verified: with the stand-down removed, exactly that happens).</summary>
    [TestMethod]
    public void PinnedInput_PassesDerivativeChecker()
    {
        var (model, p, _, v) = BuildPinnedSingle(qValue: 1.0);
        model.Options.DerivativeTest = DerivativeTest.SecondOrder;
        model.Options.CheckDerivativesForNanInf = true;

        var result = ModellingTests.SolveWithDerivativeTest(model);

        ModellingTests.AssertDerivativeTestPassed(result.DerivativeTestResult);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(3.0, result.Solution![p], 1e-6);
        Assert.AreEqual(2.0, result.Solution[v], 1e-6);
    }

    /// <summary>The same agreement across a chain of blocks, where the pinned input reaches the
    /// second block only through the first block's closure — the path that drops it has to hold
    /// transitively, not just for the block whose residual names it. Optimum only: chained blocks
    /// have a pre-existing second-order-sensitivity defect (reproducible without this change, by
    /// running the derivative checker on this same chain with q free), so asserting on the checker
    /// here would be asserting on that bug rather than on this behaviour.</summary>
    [TestMethod]
    public void PinnedInput_ChainedBlocks_AgreeWithConstant()
    {
        var (pinnedModel, pinnedP, q, pinnedV2) = BuildPinnedChain(pinQ: true);
        var pinned = pinnedModel.Solve();

        // The same model with q absent entirely — the value folded into the residual as a constant.
        var model = new Model();
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(0, 10);
        p.Start = 1;
        var v1 = model.AddVariable();
        var v2 = model.AddVariable();
        model.AddImplicitBlock([v1], [model.AddConstraint(v1 * (1 + p + 1.0) - 10 == 0)]);
        model.AddImplicitBlock([v2], [model.AddConstraint(v2 * (1 + p) - v1 == 0)]);
        model.SetObjective(Expr.Pow(v2 - 1, 2));
        var constant = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, pinned.Status);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, constant.Status);
        Assert.AreEqual(constant.Solution![p], pinned.Solution![pinnedP], 1e-6);
        Assert.AreEqual(constant.Solution[v2], pinned.Solution[pinnedV2], 1e-6);
        Assert.AreEqual(1.0, pinned.Solution[q], 1e-9, "the pinned input is still reported at its value");
    }

    /// <summary>The optimization firing, asserted on the thing it is for: the block's input closure,
    /// which is what the dense sensitivity vectors and the N² second-order cache are sized by. Read
    /// after the solve, since the rule is latched when the model is handed to Solve.
    ///
    /// IPOPT's reported Hessian non-zero count cannot serve here: it removes fixed variables from
    /// the problem itself, so it reports the reduced count either way — the same fact that makes
    /// this optimization safe also makes it invisible from outside.</summary>
    [TestMethod]
    public void PinnedInput_LeavesBlockInputClosure()
    {
        var (pinnedModel, pinnedP, pinnedQ, pinnedV) = BuildPinnedSingle(qValue: 1.0);
        pinnedModel.Solve();
        CollectionAssert.AreEquivalent(
            new[] { pinnedP }, ClosureOf(pinnedV), "pinned: q is a constant, only p is an input");
        Assert.IsNotNull(pinnedQ);

        var (freeModel, freeP, freeQ, freeV) = BuildPinnedSingle(qValue: null);
        freeModel.Solve();
        CollectionAssert.AreEquivalent(
            new[] { freeP, freeQ! }, ClosureOf(freeV), "free: both are inputs");
    }

    /// <summary>The same exclusion has to hold transitively: the chain's second block never names q
    /// itself, it inherits its inputs from the first block's closure.</summary>
    [TestMethod]
    public void PinnedInput_ChainedBlocks_ExcludePinnedTransitively()
    {
        var (model, p, q, v2) = BuildPinnedChain(pinQ: true);
        model.Solve();

        CollectionAssert.AreEquivalent(new[] { p }, ClosureOf(v2),
            "the downstream block inherits inputs from the upstream one, so q must be gone there too");
        Assert.IsFalse(ClosureOf(v2).Contains(q));
    }

    /// <summary>Both conditions that call the optimization off, each checked on its own through the
    /// closure so neither can mask the other. Under MakeConstraint and RelaxBounds a fixed variable
    /// stays a genuine unknown — IPOPT holds it in place with an equality row, or with bounds relaxed
    /// just off each other, rather than removing it from the problem — so it needs real derivatives.
    /// And while the derivative checker is on, everything keeps its columns so the checker sees the
    /// whole truth.</summary>
    [TestMethod]
    public void PinnedInput_WhenOptimizationStandsDown_StaysInClosure()
    {
        foreach (var treatment in new[] { FixedVariableTreatment.MakeConstraint, FixedVariableTreatment.RelaxBounds })
        {
            var (model, p, q, v) = BuildPinnedSingle(qValue: 1.0);
            model.Options.FixedVariableTreatment = treatment;
            model.Solve();
            CollectionAssert.AreEquivalent(new[] { p, q! }, ClosureOf(v),
                $"{treatment}: q is still a variable IPOPT decides, so it keeps its column");
        }

        var (checkedModel, checkedP, checkedQ, checkedV) = BuildPinnedSingle(qValue: 1.0);
        checkedModel.Options.DerivativeTest = DerivativeTest.SecondOrder;
        checkedModel.Solve();
        CollectionAssert.AreEquivalent(new[] { checkedP, checkedQ! }, ClosureOf(checkedV),
            "with the checker on, the pinned column stays so the check sees complete derivatives");
    }

    /// <summary>MakeConstraint end to end: the optimization is off, so IPOPT gets the pinned variable
    /// as a real unknown with real derivatives, and must still find the same optimum.</summary>
    [TestMethod]
    public void PinnedInput_MakeConstraintTreatment_SolvesCorrectly()
    {
        var (model, p, q, v) = BuildPinnedSingle(qValue: 1.0);
        model.Options.FixedVariableTreatment = FixedVariableTreatment.MakeConstraint;

        var result = model.Solve();

        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, result.Status);
        Assert.AreEqual(3.0, result.Solution![p], 1e-6);
        Assert.AreEqual(2.0, result.Solution[v], 1e-6);
        Assert.AreEqual(1.0, result.Solution[q!], 1e-6);
    }

    /// <summary>The decision-variable inputs an eliminated variable's block depends on.</summary>
    private static List<Variable> ClosureOf(Variable eliminated)
    {
        var inputs = new HashSet<Variable>();
        eliminated.Block!.CollectInputVariables(inputs);
        return [.. inputs];
    }

    /// <summary>Bounds are mutable public fields, so whether an input is pinned is only settled when
    /// the caller hands the model to Solve — and every cache sized by the input count has to be
    /// rebuilt when that answer changes. Here v = p + q is solved with q pinned (which forces p to
    /// absorb the shortfall) and then again with q freed, where the pair reaches the objective's
    /// minimum exactly. A stale closure from the first solve would leave q out of the second and
    /// return the first answer again.</summary>
    [TestMethod]
    public void PinnedInput_FreedBeforeResolve_IsPickedUpAgain()
    {
        var model = new Model();
        model.Options.PrintLevel = 0;
        var p = model.AddVariable(-100, 100);
        p.Start = 0;
        var q = model.AddVariable(4, 4);
        q.Start = 4;
        var v = model.AddVariable();
        model.AddImplicitBlock([v], [model.AddConstraint(v - p - q == 0)]);
        model.SetObjective(Expr.Pow(v - 10, 2) + Expr.Pow(p, 2));

        // q pinned at 4: minimise (p − 6)² + p² → p = 3, v = 7.
        var pinnedResult = model.Solve(updateStartValues: false);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, pinnedResult.Status);
        Assert.AreEqual(3.0, pinnedResult.Solution![p], 1e-5);
        Assert.AreEqual(7.0, pinnedResult.Solution[v], 1e-5);

        q.LowerBound = 0;
        q.UpperBound = 100;

        // q free: both terms reach zero at p = 0, q = 10.
        var freedResult = model.Solve(updateStartValues: false);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, freedResult.Status);
        Assert.AreEqual(0.0, freedResult.Solution![p], 1e-5);
        Assert.AreEqual(10.0, freedResult.Solution[q], 1e-5);
        Assert.AreEqual(10.0, freedResult.Solution[v], 1e-5);
    }

}
