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
}
