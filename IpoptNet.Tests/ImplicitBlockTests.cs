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

    /// <summary>Numerical Hessian sanity check: implicit-Euler heat decay with two parameters.
    /// Optimizer minimises Σ(T[t] - obs[t])² over (k, T0). Both parameters appear nonlinearly in T*[t]
    /// via (1 + dt·k)·T[t+1] - T[t] = dt·k·T_out. With exact Hessian wired through ImplicitBlock, IPOPT
    /// should drive dual infeasibility to KKT tolerance; with L-BFGS it doesn't. Compare the two.</summary>
    [TestMethod]
    public void ExactHessian_Converges_Where_LBFGS_Doesnt()
    {
        // Generate observations at the true k.
        const double dt = 0.5;
        const double T_out = 5.0;
        const double trueT0 = 25.0;
        const double trueK = 0.4;
        const int nSteps = 8;
        var obs = new double[nSteps + 1];
        obs[0] = trueT0;
        for (int t = 0; t < nSteps; t++)
            obs[t + 1] = (obs[t] + dt * trueK * T_out) / (1 + dt * trueK);

        ApplicationReturnStatus RunFit(HessianApproximation hessApprox)
        {
            var model = new Model();
            model.Options.HessianApproximation = hessApprox;
            model.Options.MaxIterations = 200;
            model.Options.PrintLevel = 0;
            var k = model.AddVariable(0.01, 5.0); k.Start = 1.0;
            var T = new Variable[nSteps + 1];
            T[0] = model.AddVariable(); T[0].Start = 20.0;
            for (int t = 0; t < nSteps; t++)
            {
                T[t + 1] = model.AddVariable();
                var residual = T[t + 1] + dt * k * T[t + 1] - T[t] - dt * k * T_out;
                var c = model.AddConstraint(residual == 0);
                model.AddImplicitBlock(new[] { T[t + 1] }, new[] { c });
            }
            Expr obj = 0;
            for (int t = 0; t <= nSteps; t++)
                obj += Expr.Pow(T[t] - obs[t], 2);
            model.SetObjective(obj);
            var result = model.Solve();
            return result.Status;
        }

        var exactStatus = RunFit(HessianApproximation.Exact);
        Assert.AreEqual(ApplicationReturnStatus.SolveSucceeded, exactStatus,
            $"Exact Hessian should reach KKT convergence; got {exactStatus}.");
    }

    [TestMethod]
    public void Rejects_FiniteBounds()
    {
        var model = new Model();
        var p = model.AddVariable();
        var v = model.AddVariable(0, 100);
        var c = model.AddConstraint(v - p == 0);
        Assert.ThrowsExactly<ArgumentException>(() => model.AddImplicitBlock(new[] { v }, new[] { c }));
    }

    [TestMethod]
    public void Rejects_NonUnitScale()
    {
        var model = new Model();
        var p = model.AddVariable();
        var v = model.AddVariable(double.NegativeInfinity, double.PositiveInfinity, scale: 10);
        var c = model.AddConstraint(v - p == 0);
        Assert.ThrowsExactly<ArgumentException>(() => model.AddImplicitBlock(new[] { v }, new[] { c }));
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
