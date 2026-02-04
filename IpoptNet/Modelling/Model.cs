using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace IpoptNet.Modelling;

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

    public void SetObjective(Expr objective) => _objective = objective;

    public void AddConstraint(Constraint constraint) => _constraints.Add(constraint);

    public void AddConstraint(Expr expression, double lowerBound, double upperBound) =>
        _constraints.Add(new Constraint(expression, lowerBound, upperBound));

    public ModelResult Solve(bool updateStartValues = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_objective is null)
            throw new InvalidOperationException("No objective function set");

        var n = _variables.Count;
        var m = _constraints.Count;

        // Variable bounds
        var xL = new double[n];
        var xU = new double[n];
        for (int i = 0; i < n; i++)
        {
            xL[i] = _variables[i].LowerBound;
            xU[i] = _variables[i].UpperBound;
        }

        // Constraint bounds
        var gL = new double[m];
        var gU = new double[m];
        for (int i = 0; i < m; i++)
        {
            gL[i] = _constraints[i].LowerBound;
            gU[i] = _constraints[i].UpperBound;
        }

        // Analyze sparsity
        var (jacRows, jacCols) = AnalyzeJacobianSparsity();

        // Skip Hessian computation if using limited memory approximation
        var useLimitedMemory = Options.HessianApproximation == HessianApproximation.LimitedMemory;
        var (hessRows, hessCols) = useLimitedMemory ? (Array.Empty<int>(), Array.Empty<int>()) : AnalyzeHessianSparsity();

        // Create callbacks with automatic constant matrix detection
        var evalF = CreateEvalFCallback();
        var evalGradF = _objective.IsLinear() ? CreateCachedEvalGradFCallback() : CreateEvalGradFCallback();
        var evalG = CreateEvalGCallback();
        var evalJacG = _constraints.All(c => c.Expression.IsLinear()) ? CreateCachedEvalJacGCallback(jacRows, jacCols) : CreateEvalJacGCallback(jacRows, jacCols);
        var evalH = useLimitedMemory ? CreateDummyEvalHCallback() :
                    (_objective.IsAtMostQuadratic() && _constraints.All(c => c.Expression.IsAtMostQuadratic()) ? CreateCachedEvalHCallback(hessRows, hessCols) : CreateEvalHCallback(hessRows, hessCols));

        using var solver = new IpoptSolver(
            n, xL, xU,
            m, gL, gU,
            jacRows.Length, hessRows.Length,
            evalF, evalGradF, evalG, evalJacG, evalH);

        // Auto-enable warm start if we have non-zero dual values and user hasn't explicitly set it
        if (Options.WarmStartInitPoint is null &&
            (_variables.Any(v => v.LowerBoundDualStart != 0 || v.UpperBoundDualStart != 0) ||
             _constraints.Any(c => c.DualStart != 0)))
        {
            Options.WarmStartInitPoint = true;
        }

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

        // Initialize primal variables from variable Start values, ensuring they're within bounds
        var x = new double[n];
        for (int i = 0; i < n; i++)
            x[i] = Math.Clamp(_variables[i].Start, xL[i], xU[i]);

        // Initialize dual variables
        var constraintValues = new double[m];
        var constraintMultipliers = new double[m];
        var lowerBoundMultipliers = new double[n];
        var upperBoundMultipliers = new double[n];

        for (int i = 0; i < m; i++)
            constraintMultipliers[i] = _constraints[i].DualStart;

        for (int i = 0; i < n; i++)
        {
            lowerBoundMultipliers[i] = _variables[i].LowerBoundDualStart;
            upperBoundMultipliers[i] = _variables[i].UpperBoundDualStart;
        }

        var status = solver.Solve(x, out var objValue, out var statistics, constraintValues, constraintMultipliers,
                                  lowerBoundMultipliers, upperBoundMultipliers);

        var solution = new Dictionary<Variable, double>();
        for (int i = 0; i < n; i++)
            solution[_variables[i]] = Math.Clamp(x[i], xL[i], xU[i]);

        // Update variable Start values and dual variables if requested and solution is usable
        if (updateStartValues && status is
            ApplicationReturnStatus.SolveSucceeded or
            ApplicationReturnStatus.SolvedToAcceptableLevel or
            ApplicationReturnStatus.FeasiblePointFound or
            ApplicationReturnStatus.MaximumIterationsExceeded or
            ApplicationReturnStatus.MaximumCpuTimeExceeded or
            ApplicationReturnStatus.MaximumWallTimeExceeded)
        {
            for (int i = 0; i < n; i++)
            {
                _variables[i].Start = x[i];
                _variables[i].LowerBoundDualStart = lowerBoundMultipliers[i];
                _variables[i].UpperBoundDualStart = upperBoundMultipliers[i];
            }

            for (int i = 0; i < m; i++)
                _constraints[i].DualStart = constraintMultipliers[i];
        }

        return new ModelResult(status, solution, objValue, statistics);
    }

    private (int[] rows, int[] cols) AnalyzeJacobianSparsity()
    {
        var entries = new HashSet<(int row, int col)>();
        for (int i = 0; i < _constraints.Count; i++)
        {
            var vars = new HashSet<Variable>();
            _constraints[i].Expression.CollectVariables(vars);
            foreach (var v in vars)
                entries.Add((i, v.Index));
        }

        var rows = new int[entries.Count];
        var cols = new int[entries.Count];
        var idx = 0;
        foreach (var (row, col) in entries.OrderBy(e => e.row).ThenBy(e => e.col))
        {
            rows[idx] = row;
            cols[idx] = col;
            idx++;
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
            _objective!.AccumulateGradient(x, gradF, 1.0);

            for (int i = 0; i < n; i++)
            {
                if (!IsValidNumber(gradF[i]))
                    return false;
            }

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
                var values  = new Span<double>(pValues, neleJac);
                Span<double> gradSpan = grad;

                values.Clear();

                for (int row = 0; row < m; row++)
                {
                    _constraints[row].Expression.AccumulateGradient(x, gradSpan, 1.0);

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
                if (Math.Abs(objFactor) > 1e-15)
                    _objective!.AccumulateHessian(x, hess, objFactor);

                // Constraint contributions
                for (int row = 0; row < m; row++)
                    if (Math.Abs(lambda[row]) > 1e-15)
                        _constraints[row].Expression.AccumulateHessian(x, hess, lambda[row]);

                // Copy to values array
                var values = new Span<double>(pValues, neleHess);
                values.Clear();
                foreach (var (key, value) in hess.GetEntries())
                {
                    var idx = indexMap[key];
                    values[idx] = value;
                    if (!IsValidNumber(values[idx]))
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

    private unsafe EvalGradFCallback CreateCachedEvalGradFCallback()
    {
        // Objective must be linear when this callback is used
        Debug.Assert(_objective!.IsLinear());

        // Pre-compute constant objective gradient
        var cachedGrad = new double[_variables.Count];
        _objective.AccumulateGradient(new double[_variables.Count], cachedGrad, 1.0);
        Debug.Assert(cachedGrad.All(v => IsValidNumber(v)));

        return (int n, double* pX, bool newX, double* pGradF, nint userData) =>
        {
            var gradF = new Span<double>(pGradF, n);
            for (int i = 0; i < n; i++)
                gradF[i] = cachedGrad[i];
            return true;
        };
    }

    private unsafe EvalJacGCallback CreateCachedEvalJacGCallback(int[] structRows, int[] structCols)
    {
        // All constraints must be linear when this callback is used
        Debug.Assert(_constraints.All(c => c.Expression.IsLinear()));

        var rowToEntries = new Dictionary<int, List<(int col, int idx)>>();
        for (int i = 0; i < structRows.Length; i++)
        {
            if (!rowToEntries.ContainsKey(structRows[i]))
                rowToEntries[structRows[i]] = new List<(int, int)>();
            rowToEntries[structRows[i]].Add((structCols[i], i));
        }

        // Pre-compute constant Jacobian values
        var cachedValues = new double[structRows.Length];
        var zeroX = new double[_variables.Count];
        var grad = new double[_variables.Count];
        for (int row = 0; row < _constraints.Count; row++)
        {
            Array.Clear(grad);
            _constraints[row].Expression.AccumulateGradient(zeroX, grad, 1.0);
            foreach (var (col, idx) in rowToEntries[row])
            {
                cachedValues[idx] = grad[col];
                Debug.Assert(IsValidNumber(cachedValues[idx]));
            }
        }

        return (int n, double* pVx, bool newX, int m, int neleJac, int* iRow, int* jCol, double* pValues, nint userData) =>
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
                var values = new Span<double>(pValues, neleJac);
                for (int i = 0; i < neleJac; i++)
                    values[i] = cachedValues[i];
            }
            return true;
        };
    }

    private unsafe EvalHCallback CreateCachedEvalHCallback(int[] structRows, int[] structCols)
    {
        // All constraints are at most quadratic when this callback is used
        Debug.Assert(_constraints.All(c => c.Expression.IsAtMostQuadratic()));

        // Build a map from (row, col) to index once
        var indexMap = new Dictionary<(int, int), int>();
        for (int i = 0; i < structRows.Length; i++)
            indexMap[(structRows[i], structCols[i])] = i;

        // Pre-compute objective Hessian contribution (sparse: only non-zero entries)
        var zeroX = new double[_variables.Count];
        var hess = new HessianAccumulator(_variables.Count);
        var objectiveHessEntries = new List<(int idx, double value)>();
        {
            _objective!.AccumulateHessian(zeroX, hess, 1.0);

            foreach (var (key, value) in hess.GetEntries())
                objectiveHessEntries.Add((indexMap[key], value));
        }

        // Pre-compute constraint Hessian contributions (sparse: only store non-zero entries per constraint)
        var constraintHessEntries = new List<(int idx, double value)>[_constraints.Count];
        for (int c = 0; c < _constraints.Count; c++)
        {
            hess.Clear();
            _constraints[c].Expression.AccumulateHessian(zeroX, hess, 1.0);

            if (hess.Count > 0)
            {
                constraintHessEntries[c] = new List<(int, double)>(hess.Count);
                foreach (var (key, value) in hess.GetEntries())
                    constraintHessEntries[c].Add((indexMap[key], value));
            }
        }

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
                Debug.Assert(IsValidNumber(objFactor));

                var values = new Span<double>(pValues, neleHess);

                values.Clear();
                foreach (var (idx, value) in objectiveHessEntries)
                {
                    values[idx] += objFactor * value;
                    Debug.Assert(IsValidNumber(values[idx]));
                }

                // Add constraint contributions (only iterate over non-zero entries)
                for (int c = 0; c < m; c++)
                    if (constraintHessEntries[c] != null && Math.Abs(lambda[c]) > 1e-15)
                        foreach (var (idx, value) in constraintHessEntries[c])
                        {
                            values[idx] += lambda[c] * value;
                            Debug.Assert(IsValidNumber(values[idx]));
                        }
            }
            return true;
        };
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