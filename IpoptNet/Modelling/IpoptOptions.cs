namespace IpoptNet.Modelling;

/// <summary>
/// Available linear solvers for IPOPT.
/// </summary>
public enum LinearSolver
{
    /// <summary>MUltifrontal Massively Parallel Sparse direct solver (default).</summary>
    Mumps,
    /// <summary>MA27 from Harwell Subroutines Library.</summary>
    Ma27,
    /// <summary>MA57 from Harwell Subroutines Library.</summary>
    Ma57,
    /// <summary>MA77 from Harwell Subroutines Library.</summary>
    Ma77,
    /// <summary>MA86 from Harwell Subroutines Library.</summary>
    Ma86,
    /// <summary>MA97 from Harwell Subroutines Library.</summary>
    Ma97,
    /// <summary>Pardiso from Intel Math Kernel Library (MKL).</summary>
    PardisoMkl,
    /// <summary>Pardiso from pardiso-project.org.</summary>
    PardisoProject,
    /// <summary>Watson Sparse Matrix Package.</summary>
    Wsmp,
    /// <summary>Sparse Parallel Robust Algorithms Library.</summary>
    Spral,
    /// <summary>Custom linear solver.</summary>
    Custom
}

/// <summary>
/// Hessian approximation methods.
/// </summary>
public enum HessianApproximation
{
    /// <summary>Use exact Hessian (default).</summary>
    Exact,
    /// <summary>Use limited-memory quasi-Newton approximation.</summary>
    LimitedMemory
}

/// <summary>
/// Barrier parameter update strategies.
/// </summary>
public enum MuStrategy
{
    /// <summary>Monotone decrease of barrier parameter.</summary>
    Monotone,
    /// <summary>Adaptive update of barrier parameter.</summary>
    Adaptive
}

/// <summary>
/// NLP scaling methods.
/// </summary>
public enum NlpScalingMethod
{
    /// <summary>No scaling.</summary>
    None,
    /// <summary>User-provided scaling.</summary>
    UserScaling,
    /// <summary>Gradient-based scaling.</summary>
    GradientBased,
    /// <summary>Equilibration-based scaling.</summary>
    EquilibrationBased
}

/// <summary>
/// Linear system scaling methods.
/// </summary>
public enum LinearSystemScaling
{
    /// <summary>No scaling.</summary>
    None,
    /// <summary>MC19 scaling from HSL.</summary>
    Mc19,
    /// <summary>Slack-based scaling.</summary>
    SlackBased
}

/// <summary>
/// Treatment of fixed variables.
/// </summary>
public enum FixedVariableTreatment
{
    /// <summary>Make fixed variables parameters.</summary>
    MakeParameter,
    /// <summary>Add equality constraints for fixed variables.</summary>
    MakeConstraint,
    /// <summary>Relax bounds slightly.</summary>
    RelaxBounds
}

/// <summary>
/// Derivative test options.
/// </summary>
public enum DerivativeTest
{
    /// <summary>No derivative test.</summary>
    None,
    /// <summary>Test first derivatives only.</summary>
    FirstOrder,
    /// <summary>Test second derivatives only.</summary>
    SecondOrder,
    /// <summary>Test only at starting point.</summary>
    OnlySecondOrder
}

/// <summary>
/// Options for IPOPT solver configuration.
/// </summary>
public sealed class IpoptOptions
{
    internal readonly Dictionary<string, object> Options = new();

    // Termination options
    public double? Tolerance { set => SetDouble("tol", value); }
    public int? MaxIterations { set => SetInt("max_iter", value); }
    public double? MaxWallTime { set => SetDouble("max_wall_time", value); }
    public double? MaxCpuTime { set => SetDouble("max_cpu_time", value); }
    public double? AcceptableTolerance { set => SetDouble("acceptable_tol", value); }
    public int? AcceptableIterations { set => SetInt("acceptable_iter", value); }

    // Output options
    public int? PrintLevel { set => SetInt("print_level", value); }
    public string? OutputFile { set => SetString("output_file", value); }
    public int? FilePrintLevel { set => SetInt("file_print_level", value); }
    public bool? PrintUserOptions { set => SetBool("print_user_options", value); }
    public bool? PrintOptionsDocumentation { set => SetBool("print_options_documentation", value); }

    // Algorithm options
    public LinearSolver? LinearSolverOption { set => SetEnum("linear_solver", value); }
    public HessianApproximation? HessianApproximationOption { set => SetEnum("hessian_approximation", value); }
    public MuStrategy? MuStrategyOption { set => SetEnum("mu_strategy", value); }
    public NlpScalingMethod? NlpScalingMethodOption { set => SetEnum("nlp_scaling_method", value); }
    public LinearSystemScaling? LinearSystemScalingOption { set => SetEnum("linear_system_scaling", value); }
    public FixedVariableTreatment? FixedVariableTreatmentOption { set => SetEnum("fixed_variable_treatment", value); }
    public DerivativeTest? DerivativeTestOption { set => SetEnum("derivative_test", value); }

    // Constraint/NLP options
    public double? ConstraintViolationTolerance { set => SetDouble("constr_viol_tol", value); }
    public double? DualInfeasibilityTolerance { set => SetDouble("dual_inf_tol", value); }
    public double? ComplementarityTolerance { set => SetDouble("compl_inf_tol", value); }

    // Initialization options
    public double? BoundPush { set => SetDouble("bound_push", value); }
    public double? BoundFraction { set => SetDouble("bound_frac", value); }
    public bool? WarmStartInitPoint { set => SetBool("warm_start_init_point", value); }

    // Linear solver specific options
    public double? Ma27PivotTolerance { set => SetDouble("ma27_pivtol", value); }
    public bool? Ma57AutomaticScaling { set => SetBool("ma57_automatic_scaling", value); }

    /// <summary>
    /// Set a custom string option not covered by the typed properties.
    /// </summary>
    public void SetCustomOption(string name, string value) => SetString(name, value);

    /// <summary>
    /// Set a custom integer option not covered by the typed properties.
    /// </summary>
    public void SetCustomOption(string name, int value) => SetInt(name, value);

    /// <summary>
    /// Set a custom double option not covered by the typed properties.
    /// </summary>
    public void SetCustomOption(string name, double value) => SetDouble(name, value);

    private void SetString(string name, string? value)
    {
        if (value != null)
            Options[name] = value;
        else
            Options.Remove(name);
    }

    private void SetInt(string name, int? value)
    {
        if (value.HasValue)
            Options[name] = value.Value;
        else
            Options.Remove(name);
    }

    private void SetDouble(string name, double? value)
    {
        if (value.HasValue)
            Options[name] = value.Value;
        else
            Options.Remove(name);
    }

    private void SetBool(string name, bool? value)
    {
        if (value.HasValue)
            Options[name] = value.Value ? "yes" : "no";
        else
            Options.Remove(name);
    }

    private void SetEnum<T>(string name, T? value) where T : struct, Enum
    {
        if (value.HasValue)
            Options[name] = EnumToString(value.Value);
        else
            Options.Remove(name);
    }

    private static string EnumToString<T>(T value) where T : Enum
    {
        return value switch
        {
            LinearSolver.Mumps => "mumps",
            LinearSolver.Ma27 => "ma27",
            LinearSolver.Ma57 => "ma57",
            LinearSolver.Ma77 => "ma77",
            LinearSolver.Ma86 => "ma86",
            LinearSolver.Ma97 => "ma97",
            LinearSolver.PardisoMkl => "pardisomkl",
            LinearSolver.PardisoProject => "pardiso",
            LinearSolver.Wsmp => "wsmp",
            LinearSolver.Spral => "spral",
            LinearSolver.Custom => "custom",

            HessianApproximation.Exact => "exact",
            HessianApproximation.LimitedMemory => "limited-memory",

            MuStrategy.Monotone => "monotone",
            MuStrategy.Adaptive => "adaptive",

            NlpScalingMethod.None => "none",
            NlpScalingMethod.UserScaling => "user-scaling",
            NlpScalingMethod.GradientBased => "gradient-based",
            NlpScalingMethod.EquilibrationBased => "equilibration-based",

            LinearSystemScaling.None => "none",
            LinearSystemScaling.Mc19 => "mc19",
            LinearSystemScaling.SlackBased => "slack-based",

            FixedVariableTreatment.MakeParameter => "make_parameter",
            FixedVariableTreatment.MakeConstraint => "make_constraint",
            FixedVariableTreatment.RelaxBounds => "relax_bounds",

            DerivativeTest.None => "none",
            DerivativeTest.FirstOrder => "first-order",
            DerivativeTest.SecondOrder => "second-order",
            DerivativeTest.OnlySecondOrder => "only-second-order",

            _ => throw new ArgumentException($"Unknown enum value: {value}")
        };
    }
}
