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

## Features

- **Modeling API**: Define nonlinear optimization problems using C# expressions with natural syntax
- **Automatic Differentiation**: Gradients and Hessians computed automatically via reverse-mode AD
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
model.Options.PrintLevel = 5;  // 0=no output, 5=detailed
model.Options.OutputFile = "ipopt.log";

// Configure NLP scaling
model.Options.NlpScalingMethod = NlpScalingMethod.GradientBased;

// Use custom options for advanced features
model.Options.SetCustomOption("bound_push", 0.01);
model.Options.SetCustomOption("acceptable_tol", 1e-5);

// Define and solve your problem...
var x = model.AddVariable(1, 5) { Start = 1 };
// ... rest of model setup ...
var result = model.Solve();
```

### Available Linear Solvers

- `LinearSolver.Mumps` - Default, included with IPOPT
- `LinearSolver.PardisoMkl` - Intel MKL Pardiso (included)
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
