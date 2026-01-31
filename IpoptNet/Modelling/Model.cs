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

    public ModelResult Solve(ReadOnlySpan<double> initialPoint = default)
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
        var (hessRows, hessCols) = AnalyzeHessianSparsity();

        // Create callbacks
        var evalF = CreateEvalFCallback();
        var evalGradF = CreateEvalGradFCallback();
        var evalG = CreateEvalGCallback();
        var evalJacG = CreateEvalJacGCallback(jacRows, jacCols);
        var evalH = CreateEvalHCallback(hessRows, hessCols);

        using var solver = new IpoptSolver(
            n, xL, xU,
            m, gL, gU,
            jacRows.Length, hessRows.Length,
            evalF, evalGradF, evalG, evalJacG, evalH);

        // Suppress IPOPT output by default
        if (!Options.Options.ContainsKey("print_level"))
            solver.SetOption("print_level", 0);

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

        // Initialize x
        var x = new double[n];
        if (initialPoint.Length == n)
            initialPoint.CopyTo(x);
        else
            for (int i = 0; i < n; i++)
                x[i] = Math.Clamp(0, xL[i], xU[i]);

        var constraintValues = new double[m];
        var constraintMultipliers = new double[m];
        var status = solver.Solve(x, out var objValue, constraintValues, constraintMultipliers);

        var solution = new Dictionary<Variable, double>();
        for (int i = 0; i < n; i++)
            solution[_variables[i]] = x[i];

        return new ModelResult(status, solution, objValue);
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
        // Collect all variables from objective and constraints
        var allVars = new HashSet<Variable>();
        _objective?.CollectVariables(allVars);
        foreach (var c in _constraints)
            c.Expression.CollectVariables(allVars);

        // For simplicity, assume dense lower triangular Hessian over all variables
        // A more sophisticated analysis could track actual sparsity
        var entries = new List<(int row, int col)>();
        var varIndices = allVars.Select(v => v.Index).OrderBy(i => i).ToList();
        foreach (var i in varIndices)
            foreach (var j in varIndices.Where(jj => jj <= i))
                entries.Add((i, j));

        var rows = new int[entries.Count];
        var cols = new int[entries.Count];
        for (int idx = 0; idx < entries.Count; idx++)
        {
            rows[idx] = entries[idx].row;
            cols[idx] = entries[idx].col;
        }
        return (rows, cols);
    }

    private unsafe EvalFCallback CreateEvalFCallback()
    {
        return (int n, double* x, bool newX, double* objValue, nint userData) =>
        {
            var xSpan = new ReadOnlySpan<double>(x, n);
            *objValue = _objective!.Evaluate(xSpan);
            return true;
        };
    }

    private unsafe EvalGradFCallback CreateEvalGradFCallback()
    {
        return (int n, double* x, bool newX, double* gradF, nint userData) =>
        {
            var xSpan = new ReadOnlySpan<double>(x, n);
            var gradSpan = new Span<double>(gradF, n);
            gradSpan.Clear();
            _objective!.AccumulateGradient(xSpan, gradSpan, 1.0);
            return true;
        };
    }

    private unsafe EvalGCallback CreateEvalGCallback()
    {
        return (int n, double* x, bool newX, int m, double* g, nint userData) =>
        {
            var xSpan = new ReadOnlySpan<double>(x, n);
            for (int i = 0; i < m; i++)
                g[i] = _constraints[i].Expression.Evaluate(xSpan);
            return true;
        };
    }

    private unsafe EvalJacGCallback CreateEvalJacGCallback(int[] structRows, int[] structCols)
    {
        return (int n, double* x, bool newX, int m, int neleJac, int* iRow, int* jCol, double* values, nint userData) =>
        {
            if (values == null)
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
                var xSpan = new ReadOnlySpan<double>(x, n);
                Span<double> grad = stackalloc double[n];

                // Build a map from (row, col) to index for fast lookup
                var indexMap = new Dictionary<(int, int), int>();
                for (int i = 0; i < structRows.Length; i++)
                    indexMap[(structRows[i], structCols[i])] = i;

                // Clear values
                for (int i = 0; i < neleJac; i++)
                    values[i] = 0;

                for (int row = 0; row < m; row++)
                {
                    grad.Clear();
                    _constraints[row].Expression.AccumulateGradient(xSpan, grad, 1.0);
                    for (int col = 0; col < n; col++)
                    {
                        if (indexMap.TryGetValue((row, col), out var idx))
                            values[idx] = grad[col];
                    }
                }
            }
            return true;
        };
    }

    private unsafe EvalHCallback CreateEvalHCallback(int[] structRows, int[] structCols)
    {
        return (int n, double* x, bool newX, double objFactor, int m, double* lambda, bool newLambda,
                int neleHess, int* iRow, int* jCol, double* values, nint userData) =>
        {
            if (values == null)
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
                var xSpan = new ReadOnlySpan<double>(x, n);
                Span<double> grad = stackalloc double[n];
                var hess = new HessianAccumulator(n);

                // Build a map from (row, col) to index
                var indexMap = new Dictionary<(int, int), int>();
                for (int i = 0; i < structRows.Length; i++)
                    indexMap[(structRows[i], structCols[i])] = i;

                // Clear values
                for (int i = 0; i < neleHess; i++)
                    values[i] = 0;

                // Objective contribution
                if (Math.Abs(objFactor) > 1e-15)
                    _objective!.AccumulateHessian(xSpan, grad, hess, objFactor);

                // Constraint contributions
                for (int row = 0; row < m; row++)
                {
                    if (Math.Abs(lambda[row]) > 1e-15)
                        _constraints[row].Expression.AccumulateHessian(xSpan, grad, hess, lambda[row]);
                }

                // Copy to output
                foreach (var (key, value) in hess.Entries)
                {
                    if (indexMap.TryGetValue(key, out var idx))
                        values[idx] = value;
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
    double ObjectiveValue);
