namespace IpoptNet.Modelling;

/// <summary>A variable that could be removed from the IPOPT decision vector, together with the
/// equality constraint that would define it. See <see cref="Model.FindEliminableVariables"/>.</summary>
/// <param name="Variable">The variable that would be eliminated.</param>
/// <param name="Constraint">The equality that would define it.</param>
/// <param name="Coefficient">∂constraint/∂variable — constant, by the linearity requirement. Larger
/// magnitudes are preferred when a constraint could define more than one variable, since this is the
/// 1×1 matrix the block inverts.</param>
public sealed record EliminationCandidate(Variable Variable, Constraint Constraint, double Coefficient);

public sealed partial class Model
{
    /// <summary>
    /// When true, <see cref="Solve"/> looks for variables it can move out of the IPOPT decision
    /// vector — see <see cref="FindEliminableVariables"/> — and registers them as implicit blocks
    /// before solving, as if <see cref="AddImplicitBlock"/> had been called by hand.
    ///
    /// Off by default, and deliberately so. Unlike partitioning, this is not a free win: the
    /// resulting NLP has the same optimum in exact arithmetic but is a genuinely different problem
    /// for IPOPT to walk, with different conditioning, and each eliminated variable enters the
    /// reduced problem nonlinearly through its block. Whether shrinking the decision vector pays for
    /// that is model-specific, so it is a decision to make deliberately and measure.
    ///
    /// This is an option for the solve, not an edit to the model. The restructuring exists only for
    /// the duration of the call and is undone before it returns, so a caller who turns the flag off,
    /// or who inspects the model afterwards, finds exactly what they built. Blocks a caller added by
    /// hand are of course left alone.
    /// </summary>
    public bool EnableAutomaticElimination { get; set; }

    // Below this magnitude a coefficient is treated as no coefficient at all: the block would invert
    // it, so a near-zero pivot buys a shrunken decision vector at the cost of a numerically hopeless
    // definition. Deliberately conservative — declining to eliminate is always safe.
    private const double MinimumEliminationCoefficient = 1e-6;

    // Tolerance for "the partial derivative did not move", mirroring ImplicitBlock.VerifyLinearity.
    private const double EliminationLinearityTolerance = 1e-9;

    /// <summary>
    /// Finds variables that could be removed from the IPOPT decision vector and computed from an
    /// equality constraint instead. A pair qualifies when the constraint is an equality of the form
    /// <c>expression == 0</c>, the variable's partial derivative of that expression is a non-zero
    /// constant (so the constraint is linear in it, as an implicit block requires), and the variable
    /// is unbounded — a block writes its value straight into the evaluation buffer, so a bound could
    /// only be violated silently. A non-unit <see cref="Variable.Scale"/> is fine.
    ///
    /// Each constraint defines at most one variable and each variable is defined by at most one
    /// constraint. Where a constraint could define several, the largest coefficient wins, that being
    /// the pivot the block inverts. Definitions that would form a cycle are dropped, since blocks
    /// must be registerable in dependency order.
    ///
    /// Pure: reports what is possible without changing anything. Constraints that reference an
    /// already-eliminated variable are skipped — they belong to an existing block's business.
    /// </summary>
    /// <exception cref="InvalidOperationException">No objective function has been set.</exception>
    public IReadOnlyList<EliminationCandidate> FindEliminableVariables()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_objective is null)
            throw new InvalidOperationException("No objective function set");

        var chosen = MatchConstraintsToVariables();
        return OrderByDependency(chosen);
    }

    /// <summary>One pass of greedy matching: walk the constraints in declaration order and give each
    /// the best variable still unclaimed.</summary>
    private List<EliminationCandidate> MatchConstraintsToVariables()
    {
        int totalVars = _variables.Count;
        var probe = new double[totalVars];
        var gradient = new double[totalVars];
        SeedProbePoint(probe);

        var claimed = new HashSet<Variable>();
        var chosen = new List<EliminationCandidate>();
        var vars = new HashSet<Variable>();

        foreach (var c in _constraints)
        {
            // AddImplicitBlock accepts only "expression == 0"; anything else cannot become a block.
            if (c.LowerBound != c.UpperBound || c.LowerBound != 0)
                continue;

            vars.Clear();
            using (EnterRawMode())
                c.Expression.CollectVariables(vars);

            // Leave existing blocks alone: their residuals are already spoken for, and probing a
            // constraint that reads an eliminated value would measure the wrong derivative.
            if (vars.Any(v => v.IsEliminated))
                continue;

            Variable? best = null;
            double bestCoefficient = 0;
            foreach (var v in vars)
            {
                if (claimed.Contains(v)) continue;
                if (!double.IsNegativeInfinity(v.LowerBound) || !double.IsPositiveInfinity(v.UpperBound))
                    continue;

                if (TryGetConstantCoefficient(c, v, probe, gradient) is not double coefficient)
                    continue;
                if (Math.Abs(coefficient) <= Math.Abs(bestCoefficient))
                    continue;

                best = v;
                bestCoefficient = coefficient;
            }

            if (best is null) continue;
            claimed.Add(best);
            chosen.Add(new EliminationCandidate(best, c, bestCoefficient));
        }

        // The probe prepared these expressions in raw mode. Left as-is, that preparation would win
        // over the redirect-mode Prepare the solve does later (ExprNode.Prepare is first-preparer-
        // wins), so hand them back unprepared.
        foreach (var c in _constraints)
            c.Expression.Clear();

        return chosen;
    }

    /// <summary>Fills the probe buffer with each variable's start point, using the same defaulting
    /// rule the solve applies to its initial iterate. Blocks are not run: constraints that reference
    /// an eliminated variable are excluded from consideration anyway.</summary>
    private void SeedProbePoint(double[] probe)
    {
        foreach (var v in _variables)
        {
            double lo = Math.Clamp(v.LowerBound / v.Scale, -Infinity, Infinity);
            double hi = Math.Clamp(v.UpperBound / v.Scale, -Infinity, Infinity);
            probe[v.Index] =
                v.Start.HasValue ? Math.Clamp(v.Start.Value / v.Scale, lo, hi) :
                hi == Infinity ? Math.Max(0, lo) :
                lo == -Infinity ? Math.Min(0, hi) :
                (lo + hi) * 0.5;
        }
    }

    /// <summary>∂constraint/∂variable when that derivative does not depend on the variable, null
    /// otherwise. Measured the way <see cref="ImplicitBlock.VerifyLinearity"/> measures it: evaluate
    /// the gradient with the variable at 0 and at 1 and see whether the entry moved. Raw mode, so
    /// the reading is the direct partial derivative — which is what a block residual would see.</summary>
    private double? TryGetConstantCoefficient(Constraint c, Variable v, double[] probe, double[] gradient)
    {
        double saved = probe[v.Index];
        try
        {
            using (EnterRawMode())
            {
                c.Expression.Prepare(this);

                probe[v.Index] = 0.0;
                InvalidateValueCache();
                Array.Clear(gradient);
                c.Expression.AccumulateGradient(probe, gradient);
                double atZero = gradient[v.Index];

                probe[v.Index] = 1.0;
                InvalidateValueCache();
                Array.Clear(gradient);
                c.Expression.AccumulateGradient(probe, gradient);
                double atOne = gradient[v.Index];

                if (Math.Abs(atZero - atOne) > EliminationLinearityTolerance * (1 + Math.Abs(atZero)))
                    return null;   // the constraint bends in this variable; a block cannot define it
                if (Math.Abs(atZero) < MinimumEliminationCoefficient)
                    return null;   // no usable pivot

                return atZero;
            }
        }
        finally
        {
            probe[v.Index] = saved;
            InvalidateValueCache();
        }
    }

    /// <summary>Orders the matches so that a definition comes after everything it reads, dropping
    /// any that cannot be placed. Implicit blocks must be registered in dependency order, and a
    /// cycle — v defined in terms of w while w is defined in terms of v — has no such order.
    /// Dropping a member of the cycle can free the rest, so the peel is repeated.</summary>
    private List<EliminationCandidate> OrderByDependency(List<EliminationCandidate> chosen)
    {
        var pending = chosen.ToList();
        var ordered = new List<EliminationCandidate>(pending.Count);
        var placed = new HashSet<Variable>();
        var vars = new HashSet<Variable>();

        // Which chosen variables each definition reads, itself excluded.
        var reads = new Dictionary<Variable, List<Variable>>();
        var chosenVars = pending.Select(x => x.Variable).ToHashSet();
        foreach (var candidate in pending)
        {
            vars.Clear();
            using (EnterRawMode())
                candidate.Constraint.Expression.CollectVariables(vars);
            // ReferenceEquals, not !=: Variable overloads the comparison operators to BUILD constraints.
            reads[candidate.Variable] = [.. vars.Where(v => !ReferenceEquals(v, candidate.Variable) && chosenVars.Contains(v))];
        }

        while (pending.Count > 0)
        {
            int before = ordered.Count;
            for (int i = pending.Count - 1; i >= 0; i--)
            {
                var candidate = pending[i];
                if (reads[candidate.Variable].Any(v => chosenVars.Contains(v) && !placed.Contains(v)))
                    continue;
                ordered.Add(candidate);
                placed.Add(candidate.Variable);
                pending.RemoveAt(i);
            }

            if (ordered.Count != before) continue;

            // Everything left is in, or behind, a cycle. Drop the lowest-indexed one — that frees
            // whatever only depended on it, and keeps the choice deterministic.
            var dropped = pending.MinBy(x => x.Variable.Index)!;
            pending.Remove(dropped);
            chosenVars.Remove(dropped.Variable);
        }

        // Peeling walks `pending` backwards, so restore declaration order among independent
        // definitions while keeping every dependency ahead of its dependents.
        return ordered;
    }

    /// <summary>The model state automatic elimination displaced, and how to put it back. Everything
    /// <see cref="AddImplicitBlock"/> touches is captured before the first block is registered, so
    /// undoing is a restore rather than an attempt to reverse each step.</summary>
    private sealed class EliminationScope
    {
        public required Model Model;
        public required List<Constraint> Constraints;
        public required int BlockCount;
        public required Variable[] VarsReferencedByBlocks;
        public required List<Variable> Eliminated;

        public void Undo()
        {
            foreach (var v in Eliminated)
            {
                v.Block = null;
                v.IndexInBlock = -1;
            }
            Model._implicitBlocks.RemoveRange(BlockCount, Model._implicitBlocks.Count - BlockCount);

            // Restored wholesale, and in the original order: constraint positions show up in error
            // messages, and AddImplicitBlock removed the ones it took from the middle of the list.
            Model._constraints.Clear();
            Model._constraints.AddRange(Constraints);

            Model._varsReferencedByBlocks.Clear();
            foreach (var v in VarsReferencedByBlocks)
                Model._varsReferencedByBlocks.Add(v);
        }
    }

    /// <summary>Registers the automatic eliminations and returns the means to undo them. Called by
    /// <see cref="Solve"/> when <see cref="EnableAutomaticElimination"/> is set.</summary>
    private EliminationScope ApplyAutomaticElimination()
    {
        var scope = new EliminationScope
        {
            Model = this,
            Constraints = [.. _constraints],
            BlockCount = _implicitBlocks.Count,
            VarsReferencedByBlocks = [.. _varsReferencedByBlocks],
            Eliminated = [],
        };

        try
        {
            foreach (var candidate in FindEliminableVariables())
            {
                AddImplicitBlock([candidate.Variable], [candidate.Constraint]);
                scope.Eliminated.Add(candidate.Variable);
            }
        }
        catch
        {
            // A half-applied restructuring would be worse than none: leave the model as we found it
            // and let the failure surface.
            scope.Undo();
            throw;
        }

        return scope;
    }
}
