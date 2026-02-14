using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace IpoptNet.Modelling;

[DebuggerDisplay("Variables: {_variables.Count}, Constraints: {_constraints.Count}")]
public sealed class Model : IDisposable
{
    private readonly List<Variable> _variables = new();
    private readonly List<Constraint> _constraints = new();
    private Expr? _objective;
    private bool _disposed;

    /// <summary>
    /// IPOPT solver options. Configure before calling Solve().
    /// </summary>
    public IpoptOptions Options { get; } = new();

    public Model()
    {
    }

    public Variable AddVariable(double lowerBound = double.NegativeInfinity, double upperBound = double.PositiveInfinity)
    {
        var variable = new Variable(lowerBound, upperBound) { Index = _variables.Count };
        _variables.Add(variable);
        return variable;
    }

    public Variable[] AddVariables(int x, double lowerBound, double upperBound)
    {
        var res = new Variable[x];
        for (var i = 0; i < x; i++)
            res[i] = AddVariable(lowerBound, upperBound);
        return res;
    }

    public Variable[,] AddVariables(int x, int y, double lowerBound, double upperBound)
    {
        var res = new Variable[x, y];
        for (var i = 0; i < x; i++)
            for (var j = 0; j < y; j++)
                res[i, j] = AddVariable(lowerBound, upperBound);
        return res;
    }

    public Variable[,,] AddVariables(int x, int y, int z, double lowerBound, double upperBound)
    {
        var res = new Variable[x, y, z];
        for (var i = 0; i < x; i++)
            for (var j = 0; j < y; j++)
                for (var k = 0; k < z; k++)
                    res[i, j, k] = AddVariable(lowerBound, upperBound);
        return res;
    }

    public void SetObjective(Expr objective) => _objective = objective;

    public void AddConstraint(Constraint constraint) => _constraints.Add(constraint);

    public void AddConstraint(Expr expression, double lowerBound, double upperBound) =>
        _constraints.Add(new Constraint(expression, lowerBound, upperBound));

    public override string ToString()
    {
        var sb = new StringBuilder();

        sb.AppendLine($"Variables: {_variables.Count}");
        for (int i = 0; i < _variables.Count; i++)
        {
            var v = _variables[i];
            var bounds = "";
            if (v.LowerBound == v.UpperBound)
                bounds = $" == {v.LowerBound}";
            else if (v.LowerBound > double.NegativeInfinity && v.UpperBound < double.PositiveInfinity)
                bounds = $" in [{v.LowerBound}, {v.UpperBound}]";
            else if (v.LowerBound > double.NegativeInfinity)
                bounds = $" >= {v.LowerBound}";
            else if (v.UpperBound < double.PositiveInfinity)
                bounds = $" <= {v.UpperBound}";

            var start = v.Start.HasValue ? $", start={v.Start}" : "";
            sb.AppendLine($"  x[{i}]{bounds}{start}");
        }

        sb.AppendLine();
        sb.AppendLine("Objective:");
        if (_objective is not null)
            sb.AppendLine($"  {_objective}");
        else
            sb.AppendLine("  (not set)");

        sb.AppendLine();
        sb.AppendLine($"Constraints: {_constraints.Count}");
        for (int i = 0; i < _constraints.Count; i++)
        {
            var c = _constraints[i];
            var boundsStr = "";
            if (c.LowerBound == c.UpperBound)
                boundsStr = $" == {c.LowerBound}";
            else if (c.LowerBound > double.NegativeInfinity && c.UpperBound < double.PositiveInfinity)
                boundsStr = $" in [{c.LowerBound}, {c.UpperBound}]";
            else if (c.LowerBound > double.NegativeInfinity)
                boundsStr = $" >= {c.LowerBound}";
            else if (c.UpperBound < double.PositiveInfinity)
                boundsStr = $" <= {c.UpperBound}";

            sb.AppendLine($"  Constraint[{i}]{boundsStr}: {c.Expression}");
        }

        return sb.ToString();
    }

    public ModelResult Solve(bool updateStartValues = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_objective is null)
            throw new InvalidOperationException("No objective function set");

        var n = _variables.Count;
        var m = _constraints.Count;

        const double Infinity = 1e19;

        // Variable bounds
        var xL = new double[n];
        var xU = new double[n];
        for (int i = 0; i < n; i++)
        {
            xL[i] = Math.Clamp(_variables[i].LowerBound, -Infinity, Infinity);
            xU[i] = Math.Clamp(_variables[i].UpperBound, -Infinity, Infinity);
        }

        // Constraint bounds
        var gL = new double[m];
        var gU = new double[m];
        for (int i = 0; i < m; i++)
        {
            gL[i] = _constraints[i].LowerBound;
            gU[i] = _constraints[i].UpperBound;
        }

        // Cache variables upfront to eliminate allocations during optimization
        _objective.Prepare();
        foreach (var constraint in _constraints)
            constraint.Expression.Prepare();

        // Analyze sparsity
        var (jacRows, jacCols) = AnalyzeJacobianSparsity();

        // Skip Hessian computation if using limited memory approximation
        var useLimitedMemory = Options.HessianApproximation == HessianApproximation.LimitedMemory;
        var (hessRows, hessCols) = useLimitedMemory ? (Array.Empty<int>(), Array.Empty<int>()) : AnalyzeHessianSparsity();

        // Create callbacks
        var evalF = CreateEvalFCallback();
        var evalGradF = CreateEvalGradFCallback();
        var evalG = CreateEvalGCallback();
        var evalJacG = CreateEvalJacGCallback(jacRows, jacCols);
        var evalH = useLimitedMemory ? CreateDummyEvalHCallback() : CreateEvalHCallback(hessRows, hessCols);

        ApplicationReturnStatus status;
        double objValue;
        SolveStatistics statistics;
        var constraintValues = new double[m];
        var constraintMultipliers = new double[m];
        var lowerBoundMultipliers = new double[n];
        var upperBoundMultipliers = new double[n];

        using var solver = new IpoptSolver(
            n, xL, xU,
            m, gL, gU,
            jacRows.Length, hessRows.Length,
            evalF, evalGradF, evalG, evalJacG, evalH);

        // Apply user-specified options
        foreach (var (name, value) in Options.Options)
        {
            switch (value)
            {
                case string strValue:
                    solver.SetOption(name, strValue);
                    break;
                case int intValue:
                    solver.SetOption(name, intValue);
                    break;
                case double dblValue:
                    solver.SetOption(name, dblValue);
                    break;
            }
        }

        // Auto-set constant derivative options if user hasn't explicitly set them
        // These are set directly on the solver to avoid persisting in Options across multiple solves

        // Auto-enable warm start if we have non-zero dual values and user hasn't explicitly set it
        if (Options.WarmStartInitPoint is null &&
            (_variables.Any(v => v.LowerBoundDualStart != 0 || v.UpperBoundDualStart != 0) ||
             _constraints.Any(c => c.DualStart != 0)))
        {
            solver.SetOption("warm_start_init_point", "yes");
        }

        // Auto-enable grad_f_constant if objective has constant gradients and user hasn't explicitly set it
        if (Options.GradFConstant is null && _objective.IsLinear())
            solver.SetOption("grad_f_constant", "yes");

        // Auto-enable jac_c_constant if all equality constraints have constant Jacobians
        var equalityConstraints = _constraints.Where(c => Math.Abs(c.LowerBound - c.UpperBound) < 1e-15).ToList();
        if (Options.JacCConstant is null && equalityConstraints.All(c => c.Expression.IsLinear()))
            solver.SetOption("jac_c_constant", "yes");

        // Auto-enable jac_d_constant if all inequality constraints have constant Jacobians
        var inequalityConstraints = _constraints.Where(c => Math.Abs(c.LowerBound - c.UpperBound) >= 1e-15).ToList();
        if (Options.JacDConstant is null && inequalityConstraints.All(c => c.Expression.IsLinear()))
            solver.SetOption("jac_d_constant", "yes");

        // Auto-enable hessian_constant if objective and all constraints are at most quadratic
        if (Options.HessianConstant is null && !useLimitedMemory &&
            _objective.IsAtMostQuadratic() && _constraints.All(c => c.Expression.IsLinear()))
        {
            solver.SetOption("hessian_constant", "yes");
        }

        // Initialize primal variables from variable Start values, ensuring they're within bounds
        var x = new double[n];
        for (int i = 0; i < n; i++)
            if (_variables[i].Start.HasValue)
                x[i] = Math.Clamp(_variables[i].Start!.Value, xL[i], xU[i]);
            else if (xU[i] == Infinity)
                x[i] = Math.Max(0, xL[i]);
            else if (xL[i] == -Infinity)
                x[i] = Math.Min(0, xU[i]);
            else
            {
                Debug.Assert(xU[i] != Infinity && xL[i] != -Infinity);
                x[i] = (xL[i] + xU[i]) * 0.5;
            }


        // Initialize dual variables
        for (int i = 0; i < m; i++)
            constraintMultipliers[i] = _constraints[i].DualStart;

        for (int i = 0; i < n; i++)
        {
            lowerBoundMultipliers[i] = _variables[i].LowerBoundDualStart;
            upperBoundMultipliers[i] = _variables[i].UpperBoundDualStart;
        }

        status = solver.Solve(x, out objValue, out statistics, constraintValues, constraintMultipliers,
                                  lowerBoundMultipliers, upperBoundMultipliers);

        // Build solution
        var solution = new Dictionary<Variable, double>();

        if (status is
            ApplicationReturnStatus.SolveSucceeded or
            ApplicationReturnStatus.SolvedToAcceptableLevel or
            ApplicationReturnStatus.FeasiblePointFound or
            ApplicationReturnStatus.MaximumIterationsExceeded or
            ApplicationReturnStatus.MaximumCpuTimeExceeded or
            ApplicationReturnStatus.MaximumWallTimeExceeded)
        {
            for (int i = 0; i < n; i++)
                solution[_variables[i]] = Math.Clamp(x[i], xL[i], xU[i]);

            // Update variable Start values and dual variables if requested and solution is usable
            if (updateStartValues)
            {
                for (int i = 0; i < n; i++)
                {
                    // x values are already in solution dictionary
                    _variables[i].Start = solution[_variables[i]];
                    _variables[i].LowerBoundDualStart = lowerBoundMultipliers[i];
                    _variables[i].UpperBoundDualStart = upperBoundMultipliers[i];
                }

                for (int i = 0; i < m; i++)
                    _constraints[i].DualStart = constraintMultipliers[i];
            }
        }

        // Clear cached variables to free memory after optimization
        _objective.Clear();
        foreach (var constraint in _constraints)
            constraint.Expression.Clear();

        return new ModelResult(status, solution, objValue, statistics);
    }

    private (int[] rows, int[] cols) AnalyzeJacobianSparsity()
    {
        var entries = new HashSet<(int row, int col)>();
        var vars = new HashSet<Variable>();

        for (int i = 0; i < _constraints.Count; i++)
        {
            vars.Clear();
            _constraints[i].Expression.CollectVariables(vars);
            foreach (var v in vars)
                entries.Add((i, v.Index));
        }

        var entriesArray = new (int row, int col)[entries.Count];
        entries.CopyTo(entriesArray);
        Array.Sort(entriesArray, (a, b) =>
        {
            int cmp = a.row.CompareTo(b.row);
            return cmp != 0 ? cmp : a.col.CompareTo(b.col);
        });

        var rows = new int[entriesArray.Length];
        var cols = new int[entriesArray.Length];
        for (int i = 0; i < entriesArray.Length; i++)
        {
            rows[i] = entriesArray[i].row;
            cols[i] = entriesArray[i].col;
        }
        return (rows, cols);
    }

    private (int[] rows, int[] cols) AnalyzeHessianSparsity()
    {
        var entries = new HashSet<(int row, int col)>();
        _objective?.CollectHessianSparsity(entries);
        foreach (var c in _constraints)
            c.Expression.CollectHessianSparsity(entries);

        var sortedEntries = entries.OrderBy(e => e.row).ThenBy(e => e.col).ToList();

        var rows = new int[sortedEntries.Count];
        var cols = new int[sortedEntries.Count];
        for (int i = 0; i < sortedEntries.Count; i++)
        {
            rows[i] = sortedEntries[i].row;
            cols[i] = sortedEntries[i].col;
        }
        return (rows, cols);
    }

    private unsafe EvalFCallback CreateEvalFCallback()
    {
        return (int n, double* pX, bool newX, double* objValue, nint userData) =>
        {
            var x = new ReadOnlySpan<double>(pX, n);
            *objValue = _objective!.Evaluate(x);
            return IsValidNumber(*objValue);
        };
    }

    private unsafe EvalGradFCallback CreateEvalGradFCallback()
    {
        return (int n, double* pX, bool newX, double* pGradF, nint userData) =>
        {
            var x = new ReadOnlySpan<double>(pX, n);
            var gradF = new Span<double>(pGradF, n);
            gradF.Clear();
            _objective!.AccumulateGradient(x, gradF);

            for (int i = 0; i < n; i++)
                if (!IsValidNumber(gradF[i]))
                    return false;

            return true;
        };
    }

    public static bool IsValidNumber(double v) => !double.IsInfinity(v) && !double.IsNaN(v);

    private unsafe EvalGCallback CreateEvalGCallback()
    {
        return (int n, double* pX, bool newX, int m, double* pG, nint userData) =>
        {
            var x = new ReadOnlySpan<double>(pX, n);
            var g = new Span<double>(pG, m);
            for (int i = 0; i < m; i++)
            {
                g[i] = _constraints[i].Expression.Evaluate(x);
                if (!IsValidNumber(g[i]))
                    return false;
            }
            return true;
        };
    }

    private unsafe EvalJacGCallback CreateEvalJacGCallback(int[] structRows, int[] structCols)
    {
        // Build a map from row -> list of (col, valueIndex) for only the sparse entries
        var rowToEntries = new Dictionary<int, List<(int col, int idx)>>();
        for (int i = 0; i < structRows.Length; i++)
        {
            if (!rowToEntries.ContainsKey(structRows[i]))
                rowToEntries[structRows[i]] = new List<(int, int)>();
            rowToEntries[structRows[i]].Add((structCols[i], i));
        }

        // Allocate gradient buffer once and reuse it
        var grad = new double[_variables.Count];

        return (int n, double* pX, bool newX, int m, int neleJac, int* iRow, int* jCol, double* pValues, nint userData) =>
        {
            if (pValues == null)
            {
                // Return sparsity structure
                for (int i = 0; i < structRows.Length; i++)
                {
                    iRow[i] = structRows[i];
                    jCol[i] = structCols[i];
                }
            }
            else
            {
                // Compute values
                var x = new ReadOnlySpan<double>(pX, n);
                var values = new Span<double>(pValues, neleJac);
                Span<double> gradSpan = grad;

                values.Clear();

                for (int row = 0; row < m; row++)
                {
                    _constraints[row].Expression.AccumulateGradient(x, gradSpan);

                    foreach (var (col, idx) in rowToEntries[row])
                    {
                        values[idx] = gradSpan[col];
                        if (!IsValidNumber(values[idx]))
                            return false;
                        gradSpan[col] = 0;  // Clear the sparse entries we used
                    }
                }
            }
            return true;
        };
    }

    private unsafe EvalHCallback CreateEvalHCallback(int[] structRows, int[] structCols)
    {
        // Build a map from (row, col) to index once
        var indexMap = new Dictionary<(int, int), int>();
        for (int i = 0; i < structRows.Length; i++)
            indexMap[(structRows[i], structCols[i])] = i;

        var hess = new HessianAccumulator(_variables.Count);

        return (int n, double* pX, bool newX, double objFactor, int m, double* lambda, bool newLambda,
                int neleHess, int* iRow, int* jCol, double* pValues, nint userData) =>
        {
            if (pValues == null)
            {
                // Return sparsity structure
                for (int i = 0; i < structRows.Length; i++)
                {
                    iRow[i] = structRows[i];
                    jCol[i] = structCols[i];
                }
            }
            else
            {
                var x = new ReadOnlySpan<double>(pX, n);
                hess.Clear();

                // Objective contribution
                _objective!.AccumulateHessian(x, hess, objFactor);

                // Constraint contributions
                for (int row = 0; row < m; row++)
                    _constraints[row].Expression.AccumulateHessian(x, hess, lambda[row]);

                // Copy to values array
                var values = new Span<double>(pValues, neleHess);
                values.Clear();
                for (int i = 0; i < structRows.Length; i++)
                {
                    values[i] = hess.Get(structRows[i], structCols[i]);
                    if (!IsValidNumber(values[i]))
                        return false;
                }
            }
            return true;
        };
    }

    private unsafe EvalHCallback CreateDummyEvalHCallback()
    {
        return (int n, double* pX, bool newX, double objFactor, int m, double* lambda, bool newLambda,
                int neleHess, int* iRow, int* jCol, double* pValues, nint userData) => false;
    }

    public void Dispose()
    {
        _disposed = true;
    }
}

public sealed record ModelResult(
    ApplicationReturnStatus Status,
    IReadOnlyDictionary<Variable, double> Solution,
    double ObjectiveValue,
    SolveStatistics Statistics);