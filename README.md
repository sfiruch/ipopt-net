# IpoptNet

![.NET](https://github.com/sfiruch/ipopt-net/actions/workflows/dotnet.yml/badge.svg)
[![NuGet](https://img.shields.io/nuget/v/ipopt-net.svg)](https://www.nuget.org/packages/ipopt-net/)

A modern .NET interface for [IPOPT (Interior Point OPTimizer)](https://coin-or.github.io/Ipopt/), a software library for large-scale nonlinear optimization. This library provides both a high-level modeling API with automatic differentiation and a low-level native wrapper.

## Installation

Install the package via NuGet:

```bash
dotnet add package ipopt-net
```

The package includes native binaries for:
- **Windows** (x64)
- **Linux** (x64)

## Features

- **Modeling API**: Define nonlinear optimization problems using C# expressions with natural syntax
- **Automatic Differentiation**: Gradients and Hessians computed automatically via reverse-mode AD
- **Intelligent Matrix Caching**: Automatically detects and pre-computes constant matrices for LP/QP/QCP problems
- **Automatic Partitioning**: Detects models that split into independent sub-problems and solves each separately
- **High-level Wrapper**: Clean, disposable `IpoptSolver` class for direct API access
- **Native Performance**: Uses .NET 10 `LibraryImport` for efficient C API calls
- **Expression Support**: Arithmetic, trigonometric, exponential, logarithmic, and power operations
- **Flexible Constraints**: Equality, inequality, and bound constraints

## Quick Start (Modeling API)

The modeling API allows you to define optimization problems with automatic differentiation:

```csharp
using IpoptNet.Modelling;

// Create a model
var model = new Model();

// Configure IPOPT (optional)
model.Options.LinearSolver = LinearSolver.PardisoMkl;
model.Options.HessianApproximation = HessianApproximation.LimitedMemory;

// Add variables with bounds and optional initial guesses
var x = model.AddVariable(1, 5);
var y = model.AddVariable(1, 5) { Start = 3.7 };
var z = model.AddVariable(1, 5);
var w = model.AddVariable(1, 5);

// Set objective: minimize x*w*(x+y+z) + z (expressions can be built incrementally)
var expr = x * (x + y + z);
expr *= w;
model.SetObjective(expr + z);

// Add constraints
model.AddConstraint(x * y * z * w >= 25);
model.AddConstraint(x*x + y*y + z*z + w*w == 40);

// Solve
var result = model.Solve();

if (result.Status == ApplicationReturnStatus.SolveSucceeded)
{
    Console.WriteLine($"x = {result.Solution[x]:F3}");
    Console.WriteLine($"y = {result.Solution[y]:F3}");
    Console.WriteLine($"z = {result.Solution[z]:F3}");
    Console.WriteLine($"w = {result.Solution[w]:F3}");
    Console.WriteLine($"Objective = {result.ObjectiveValue:F3}");
}
```

**Output:**
```
x = 1.000
y = 4.743
z = 3.821
w = 1.379
Objective = 17.014
```

## Supported Operations

The expression system supports:

- **Arithmetic**: `+`, `-`, `*`, `/`, unary `-`
- **Power**: `Expr.Pow(x, n)`, `Expr.Sqrt(x)`
- **Trigonometric**: `Expr.Sin(x)`, `Expr.Cos(x)`, `Expr.Tan(x)`
- **Exponential/Log**: `Expr.Exp(x)`, `Expr.Log(x)`
- **Constraints**: `>=`, `<=`, `==`

## Partitioning

Many models decompose into independent sub-problems that share no variable through any constraint,
implicit block, or objective term. Because IPOPT's linear-algebra cost grows superlinearly with
problem size, solving each sub-problem separately is both exact and considerably cheaper.

This is **on by default** — no configuration needed. `result.Partitions` exposes the individual
sub-problems when a model does decompose:

```csharp
var model = new Model();
// ... build the model ...
var result = model.Solve();

foreach (var partition in result.Partitions)
    Console.WriteLine($"{partition.Status}: {partition.ObjectiveValue}");
```

Set `Model.EnablePartitioning = false` to force a single whole-model solve — the pre-partitioning
behaviour, byte-identical, skipping the decomposition analysis entirely.

`Status`, `Solution`, `ObjectiveValue` and `Statistics` on the returned `ModelResult` are
model-level aggregates, so a partitioned solve reads the same as an unpartitioned one; the
individual sub-problem results are on `Partitions`. Every partition is always attempted, so one
failing sub-problem never suppresses the others.

Sub-problems are solved **smallest first** — by variables + constraints, the dimension of the KKT
system IPOPT factorises. Under a model-wide time budget that maximises how many finish before the
deadline, and it fills `BestIterate` with the cheap wins instead of leaving it empty behind one
slow partition. Ties break on the smallest `Variable.Index`, so the order is deterministic. Eliminated variables are not
counted — the solver does not decide them.

A partition whose variables are *all* eliminated by implicit blocks is reported by
`AnalyzePartitions` but never handed to IPOPT: there is nothing left to decide, and a
zero-variable NLP cannot be created. Its variables come from its blocks, and its objective
slice — constant by construction — joins the model total.

### Constant constraints

A constraint that references no decision variable — most often a bound on a variable an implicit
block pins to a constant — cannot be given to IPOPT at all: its Jacobian row is empty, which the C
API rejects. Such constraints are evaluated once before the solve and then left out of the problem.
If one cannot hold, `Solve()` throws naming the fixed value and the bound it misses, rather than
leaving you to diagnose a bare infeasible status. Note this applies only when the block has *no*
decision inputs; if it has any, the constraint resolves to them, gets a real Jacobian row, and IPOPT
judges its feasibility as usual.

Inspect the decomposition without solving — this works regardless of the flag and needs no solve
state:

```csharp
var partitioning = model.AnalyzePartitions();
Console.WriteLine(partitioning);            // one line per partition
if (partitioning.IsTrivial) { /* the model does not decompose */ }
```

When the model does not decompose, `Solve()` takes the ordinary single-solve path, so results are
bit-identical to disabling partitioning.

### Iteration callback (breaking change)

`Model.IntermediateCallback` gained a second parameter so an iteration can be attributed to a
sub-problem:

```csharp
// before: Func<SolveStatistics, bool>
model.IntermediateCallback = (stats, partition) =>
{
    // stats always describes the whole model: ObjectiveValue accounts for partitions already
    // solved, the one currently iterating, and the ones not yet started; IterationCount is
    // cumulative. So best-so-far tracking needs no changes.
    // partition.Index / .Count identify the sub-problem; partition.LocalStatistics has the raw
    // per-partition numbers. With partitioning off these are 0 and 1.
    return !cancelled;
};
```

### Limits and budgets

Iteration and time limits are treated differently, on purpose:

| Option | Scope | Why |
|---|---|---|
| `MaxIterations` | **per partition** | It guards against one sub-problem spinning forever. Dividing it would make a later partition fail merely for having followed a hard one. So total iterations can exceed the limit. |
| `MaxWallTime`, `MaxCpuTime` | **model-wide** | These are deadlines. Each partition is handed what remains of the budget, so N partitions cannot take N times as long as you allowed. |

Elapsed wall time is measured exactly. Elapsed CPU time is taken from the process total, which
over-counts when other threads in your application are busy — it therefore errs toward stopping
sooner, never toward overrunning the budget.

Two other things change when the model actually decomposes:

- `OutputFile` receives the concatenation of N IPOPT runs (`file_append` is set automatically for
  the second and later partitions unless you set it yourself).
- *Inert* variables — those appearing in no constraint, no implicit block and no objective term —
  never reach IPOPT. Nothing optimises or constrains such a variable, so a solve would only hand back
  wherever the barrier drifted it; that value depends on the surrounding problem's iteration count
  and so cannot match an unpartitioned solve either way. They are resolved directly from their start
  point instead — an explicit `Start` clamped to bounds, otherwise the same bound-derived default
  IPOPT would have used — which at least makes it deterministic. They contribute no entry to
  `result.Partitions`, and variables the model actually references are unaffected.

## Best Iterate

IPOPT returns its **final** iterate, which is not always its best one. A run that ends on
`MaximumIterationsExceeded`, `RestorationFailed`, or a caller-requested stop can finish somewhere
worse than it passed through earlier. Every solve therefore records the best point it saw:

```csharp
var result = model.Solve();

var best = result.BestIterate;
if (best is not null && best.IsFeasible)
    Console.WriteLine($"best objective {best.ObjectiveValue} at iteration {best.IterationCount}");
```

`BestIterate.Solution` covers every variable, implicit-block-eliminated ones included, and under
partitioning it is the whole model's — no partition bookkeeping required.

**"Best" is feasibility-first**, not lowest-objective: the lowest-objective iterate whose constraint
violation is within `ConstraintViolationTolerance`, falling back to the least-infeasible point (with
`IsFeasible` false) when nothing feasible was ever seen. That distinction is not academic. Minimising
`x` on the unit circle, stopped after 7 iterations, IPOPT's final iterate reports an objective of
**-1.977** — below the true optimum of -1, because it sits well outside the circle with a violation
of 5.8. The snapshot instead holds a point at objective 0.900 with a violation of 0.004. A tracker
that merely minimised the objective would have handed back the nonsense point.

Two caveats. Restoration-phase iterates are skipped, their objective belonging to IPOPT's internal
restoration problem rather than yours. And the snapshot can be *marginally less* feasible than
`Solution`: once two points are both inside the tolerance they count as equally feasible and the
lower objective wins, so a converged run may report a snapshot at the tolerance edge whose objective
is a hair under the true optimum.

If you want the raw iterate yourself, `IpoptSolver.TryGetCurrentIterate` exposes it, callable only
from inside an intermediate callback.

## Automatic Elimination

A variable defined by an equality it appears in linearly can be moved out of IPOPT's decision vector
and computed from that equality instead — the `AddImplicitBlock` mechanism, found rather than
declared. `Model.FindEliminableVariables()` reports what qualifies without changing anything:

```csharp
foreach (var c in model.FindEliminableVariables())
    Console.WriteLine($"x[{c.Variable.Index}] could be defined by its constraint (coefficient {c.Coefficient})");

model.EnableAutomaticElimination = true;   // off by default
```

A pair qualifies when the constraint is an equality of the form `expression == 0`, the variable's
partial derivative of it is a non-zero constant, and the variable is **unbounded** — a block writes
its value straight into the evaluation buffer, so a bound could only be violated silently. A
non-unit `Scale` is fine. Each constraint defines at most one variable and vice versa; where a
constraint could define several, the largest coefficient wins, that being the pivot the block
inverts. Definitions that would form a cycle are dropped, since blocks must be registered in
dependency order.

**This is off by default and should stay a deliberate choice.** Unlike partitioning it is not a free
win: the reduced problem has the same optimum in exact arithmetic but is a different problem for
IPOPT to walk, with different conditioning, and each eliminated variable enters it nonlinearly
through its block. Measure before adopting it.

The flag is an option for the solve, not an edit to your model: the restructuring exists only for the
duration of the `Solve()` call and is undone before it returns. Turn the flag off and the next solve
sees exactly what you built; inspect the model afterwards and you find your own constraints, not
blocks. Blocks you added by hand are left alone throughout.

## Performance Optimization

The solver automatically detects problem structure and optimizes matrix computations:

### Constant Matrix Detection

For certain problem types, derivative matrices remain constant throughout the solution process. The library automatically detects these cases and pre-computes matrices once:

| Problem Type | Constant Matrices | Description |
|-------------|------------------|-------------|
| **Linear Programming (LP)** | Gradient, Jacobian | All derivatives are constant coefficients |
| **Quadratic Programming (QP)** | Jacobian, Hessian | Linear constraints have constant gradients; quadratic terms have constant second derivatives |
| **Quadratically Constrained (QCP)** | Hessian contributions | Quadratic constraints contribute constant Hessian terms |

**Example - Linear Program:**
```csharp
var model = new Model();
var x = model.AddVariable(0, 10);
var y = model.AddVariable(0, 10);

// Linear objective and constraints - matrices computed once
model.SetObjective(2*x + 3*y);
model.AddConstraint(x + 2*y <= 10);
model.AddConstraint(3*x + y <= 12);

var result = model.Solve();
```

**Example - Quadratic Program:**
```csharp
var model = new Model();
var x = model.AddVariable();
var y = model.AddVariable();

// Quadratic objective, linear constraints - Hessian and Jacobian computed once
model.SetObjective(x*x + y*y - 4*x - 6*y);
model.AddConstraint(x + y <= 5);

var result = model.Solve();
```

This optimization is completely automatic - no code changes required. The solver analyzes the expression structure and applies the appropriate strategy.

## More Examples

### Rosenbrock Function (Unconstrained)

```csharp
var model = new Model();
var x = model.AddVariable();
var y = model.AddVariable();

// Minimize (1-x)^2 + 100*(y-x^2)^2
model.SetObjective(Expr.Pow(1 - x, 2) + 100 * Expr.Pow(y - x*x, 2));

var result = model.Solve();
// Converges to x=1, y=1
```

### Constrained Optimization

```csharp
var model = new Model();
var x = model.AddVariable();
var y = model.AddVariable();

// Minimize x^2 + y^2
model.SetObjective(x*x + y*y);

// Subject to x + y = 4
model.AddConstraint(x + y == 4);

var result = model.Solve();
// Solution: x=2, y=2, objective=8
```

### Trigonometric Optimization

```csharp
var model = new Model();
var x = model.AddVariable(-Math.PI, Math.PI);

// Minimize -sin(x)
model.SetObjective(-Expr.Sin(x));

var result = model.Solve();
// Converges to x=π/2
```

## Configuring IPOPT Options

The modeling API exposes all IPOPT configuration options through a strongly-typed API with enums:

```csharp
var model = new Model();

// Configure solver options using enums (type-safe with IntelliSense)
model.Options.LinearSolver = LinearSolver.PardisoMkl;  // Use Intel MKL Pardiso
model.Options.HessianApproximation = HessianApproximation.Exact;
model.Options.MuStrategy = MuStrategy.Adaptive;

// Configure termination criteria
model.Options.Tolerance = 1e-7;
model.Options.MaxIterations = 100;
model.Options.MaxWallTime = 60.0;  // seconds

// Configure output verbosity
model.Options.PrintLevel = 5;  // 0=no output, 5=default, 12=verbose
model.Options.OutputFile = "ipopt.log";

// Configure NLP scaling
model.Options.NlpScalingMethod = NlpScalingMethod.GradientBased;

// Use custom options for advanced features
model.Options.SetCustomOption("bound_push", 0.01);
model.Options.SetCustomOption("acceptable_tol", 1e-5);

// Define and solve your problem...
// ...
var result = model.Solve();
```

### Available Linear Solvers

- `LinearSolver.Mumps` - Default, included with IPOPT
- `LinearSolver.PardisoMkl` - Intel MKL Pardiso, included with IPOPT
- `LinearSolver.PardisoProject` - Pardiso from pardiso-project.org (often faster, requires external library)
- `LinearSolver.Ma27`, `Ma57`, `Ma77`, `Ma86`, `Ma97` - HSL solvers (require external library)
- `LinearSolver.Wsmp` - Watson Sparse Matrix Package (requires external library)
- `LinearSolver.Spral` - Sparse Parallel Robust Algorithms Library (requires external library)

### Common Options

- **Termination:** `Tolerance`, `MaxIterations`, `MaxWallTime`, `MaxCpuTime`
- **Output:** `PrintLevel`, `OutputFile`, `PrintUserOptions`
- **Algorithm:** `LinearSolver`, `HessianApproximation`, `MuStrategy`
- **Scaling:** `NlpScalingMethod`, `LinearSystemScaling`
- **Tolerances:** `ConstraintViolationTolerance`, `DualInfeasibilityTolerance`

## Low-level API

For advanced users who want direct control over the IPOPT solver:

```csharp
using IpoptNet;

// Define callback functions
EvalFCallback evalF = (n, x, newX, objValue, userData) =>
{
    *objValue = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];

    // Note: If a callback cannot be evaluated at a given point (e.g. division by zero), 
    // it should return false. IPOPT will then attempt to backtrack to a valid point.
    // If it cannot recover, the solve will terminate with InvalidNumberDetected.
    return true;
};

// Define gradient, constraint, Jacobian, and Hessian callbacks...

// Create solver
using var solver = new IpoptSolver(
    n: 4, xL, xU,
    m: 2, gL, gU,
    jacobianNonZeros, hessianNonZeros,
    evalF, evalGradF, evalG, evalJacG, evalH);

// Set options
solver.SetOption("print_level", 5);
solver.SetOption("tol", 1e-7);

// Solve
var x = new double[] { 1, 5, 5, 1 };
var status = solver.Solve(x, out var objValue);
```

## Problem Formulation

IPOPT solves nonlinear optimization problems of the form:

```
minimize    f(x)
subject to  g_L ≤ g(x) ≤ g_U
            x_L ≤ x ≤ x_U
```

where:
- `f(x)` is the objective function
- `g(x)` are constraint functions
- `x` are the optimization variables
- Bounds can be infinite for unconstrained dimensions

## References

- **IPOPT Project**: [https://coin-or.github.io/Ipopt/](https://coin-or.github.io/Ipopt/)
- **IPOPT Documentation**: [https://coin-or.github.io/Ipopt/DOCUMENTATION.html](https://coin-or.github.io/Ipopt/DOCUMENTATION.html)
- **IPOPT Paper**: Wächter & Biegler (2006), "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming"

## License

This .NET wrapper is provided as-is. IPOPT itself is released under the Eclipse Public License (EPL).

## Acknowledgments

IPOPT is developed and maintained by the COIN-OR project. This wrapper provides a convenient .NET interface with automatic differentiation capabilities.

The native binaries bundled with this package include statically-linked Intel oneAPI Math Kernel Library (MKL) components (libmkl_intel_lp64, libmkl_sequential, libmkl_core) redistributed under the [Intel Simplified Software License](INTEL-MKL-LICENSE.txt).
