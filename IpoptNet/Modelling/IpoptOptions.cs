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

    // AOT-friendly mappings (single source of truth)
    private static readonly Dictionary<Modelling.LinearSolver, string> LinearSolverMap = new()
    {
        [Modelling.LinearSolver.Mumps] = "mumps",
        [Modelling.LinearSolver.Ma27] = "ma27",
        [Modelling.LinearSolver.Ma57] = "ma57",
        [Modelling.LinearSolver.Ma77] = "ma77",
        [Modelling.LinearSolver.Ma86] = "ma86",
        [Modelling.LinearSolver.Ma97] = "ma97",
        [Modelling.LinearSolver.PardisoMkl] = "pardisomkl",
        [Modelling.LinearSolver.PardisoProject] = "pardiso",
        [Modelling.LinearSolver.Wsmp] = "wsmp",
        [Modelling.LinearSolver.Spral] = "spral",
        [Modelling.LinearSolver.Custom] = "custom"
    };

    private static readonly Dictionary<Modelling.HessianApproximation, string> HessianApproximationMap = new()
    {
        [Modelling.HessianApproximation.Exact] = "exact",
        [Modelling.HessianApproximation.LimitedMemory] = "limited-memory"
    };

    private static readonly Dictionary<Modelling.MuStrategy, string> MuStrategyMap = new()
    {
        [Modelling.MuStrategy.Monotone] = "monotone",
        [Modelling.MuStrategy.Adaptive] = "adaptive"
    };

    private static readonly Dictionary<Modelling.NlpScalingMethod, string> NlpScalingMethodMap = new()
    {
        [Modelling.NlpScalingMethod.None] = "none",
        [Modelling.NlpScalingMethod.UserScaling] = "user-scaling",
        [Modelling.NlpScalingMethod.GradientBased] = "gradient-based",
        [Modelling.NlpScalingMethod.EquilibrationBased] = "equilibration-based"
    };

    private static readonly Dictionary<Modelling.LinearSystemScaling, string> LinearSystemScalingMap = new()
    {
        [Modelling.LinearSystemScaling.None] = "none",
        [Modelling.LinearSystemScaling.Mc19] = "mc19",
        [Modelling.LinearSystemScaling.SlackBased] = "slack-based"
    };

    private static readonly Dictionary<Modelling.FixedVariableTreatment, string> FixedVariableTreatmentMap = new()
    {
        [Modelling.FixedVariableTreatment.MakeParameter] = "make_parameter",
        [Modelling.FixedVariableTreatment.MakeConstraint] = "make_constraint",
        [Modelling.FixedVariableTreatment.RelaxBounds] = "relax_bounds"
    };

    private static readonly Dictionary<Modelling.DerivativeTest, string> DerivativeTestMap = new()
    {
        [Modelling.DerivativeTest.None] = "none",
        [Modelling.DerivativeTest.FirstOrder] = "first-order",
        [Modelling.DerivativeTest.SecondOrder] = "second-order",
        [Modelling.DerivativeTest.OnlySecondOrder] = "only-second-order"
    };

    // Termination options
    public double? Tolerance { get => GetDouble("tol"); set => SetDouble("tol", value); }
    public int? MaxIterations { get => GetInt("max_iter"); set => SetInt("max_iter", value); }
    public double? MaxWallTime { get => GetDouble("max_wall_time"); set => SetDouble("max_wall_time", value); }
    public double? MaxCpuTime { get => GetDouble("max_cpu_time"); set => SetDouble("max_cpu_time", value); }
    public double? AcceptableTolerance { get => GetDouble("acceptable_tol"); set => SetDouble("acceptable_tol", value); }
    public int? AcceptableIterations { get => GetInt("acceptable_iter"); set => SetInt("acceptable_iter", value); }

    // Output options
    public int? PrintLevel { get => GetInt("print_level"); set => SetInt("print_level", value); }
    public string? OutputFile { get => GetString("output_file"); set => SetString("output_file", value); }
    public int? FilePrintLevel { get => GetInt("file_print_level"); set => SetInt("file_print_level", value); }
    public bool? PrintUserOptions { get => GetBool("print_user_options"); set => SetBool("print_user_options", value); }
    public bool? PrintOptionsDocumentation { get => GetBool("print_options_documentation"); set => SetBool("print_options_documentation", value); }

    // Algorithm options
    public LinearSolver? LinearSolver
    {
        get => GetEnum<LinearSolver>("linear_solver");
        set => SetEnum("linear_solver", value);
    }

    public HessianApproximation? HessianApproximation
    {
        get => GetEnum<HessianApproximation>("hessian_approximation");
        set => SetEnum("hessian_approximation", value);
    }

    public MuStrategy? MuStrategy
    {
        get => GetEnum<MuStrategy>("mu_strategy");
        set => SetEnum("mu_strategy", value);
    }

    public NlpScalingMethod? NlpScalingMethod
    {
        get => GetEnum<NlpScalingMethod>("nlp_scaling_method");
        set => SetEnum("nlp_scaling_method", value);
    }

    public LinearSystemScaling? LinearSystemScaling
    {
        get => GetEnum<LinearSystemScaling>("linear_system_scaling");
        set => SetEnum("linear_system_scaling", value);
    }

    public FixedVariableTreatment? FixedVariableTreatment
    {
        get => GetEnum<FixedVariableTreatment>("fixed_variable_treatment");
        set => SetEnum("fixed_variable_treatment", value);
    }

    public DerivativeTest? DerivativeTest
    {
        get => GetEnum<DerivativeTest>("derivative_test");
        set => SetEnum("derivative_test", value);
    }

    public double? DerivativeTestPerturbation { get => GetDouble("derivative_test_perturbation"); set => SetDouble("derivative_test_perturbation", value); }
    public double? DerivativeTestTolerance { get => GetDouble("derivative_test_tol"); set => SetDouble("derivative_test_tol", value); }

    // Constraint/NLP options
    public double? ConstraintViolationTolerance { get => GetDouble("constr_viol_tol"); set => SetDouble("constr_viol_tol", value); }
    public double? DualInfeasibilityTolerance { get => GetDouble("dual_inf_tol"); set => SetDouble("dual_inf_tol", value); }
    public double? ComplementarityTolerance { get => GetDouble("compl_inf_tol"); set => SetDouble("compl_inf_tol", value); }

    // Initialization options
    public double? BoundPush { get => GetDouble("bound_push"); set => SetDouble("bound_push", value); }
    public double? BoundFraction { get => GetDouble("bound_frac"); set => SetDouble("bound_frac", value); }

    // Warm start options
    public bool? WarmStartInitPoint { get => GetBool("warm_start_init_point"); set => SetBool("warm_start_init_point", value); }
    public double? WarmStartBoundPush { get => GetDouble("warm_start_bound_push"); set => SetDouble("warm_start_bound_push", value); }
    public double? WarmStartBoundFrac { get => GetDouble("warm_start_bound_frac"); set => SetDouble("warm_start_bound_frac", value); }
    public double? WarmStartMultBoundPush { get => GetDouble("warm_start_mult_bound_push"); set => SetDouble("warm_start_mult_bound_push", value); }
    public double? WarmStartSlackBoundPush { get => GetDouble("warm_start_slack_bound_push"); set => SetDouble("warm_start_slack_bound_push", value); }
    public double? WarmStartSlackBoundFrac { get => GetDouble("warm_start_slack_bound_frac"); set => SetDouble("warm_start_slack_bound_frac", value); }

    // Linear solver specific options
    public double? Ma27PivotTolerance { get => GetDouble("ma27_pivtol"); set => SetDouble("ma27_pivtol", value); }
    public bool? Ma57AutomaticScaling { get => GetBool("ma57_automatic_scaling"); set => SetBool("ma57_automatic_scaling", value); }

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

    private string? GetString(string name) => Options.TryGetValue(name, out var val) ? val as string : null;
    private int? GetInt(string name) => Options.TryGetValue(name, out var val) && val is int i ? i : null;
    private double? GetDouble(string name) => Options.TryGetValue(name, out var val) && val is double d ? d : null;

    private bool? GetBool(string name)
    {
        if (Options.TryGetValue(name, out var val) && val is string s)
            return s.Equals("yes", StringComparison.OrdinalIgnoreCase);
        return null;
    }

    private T? GetEnum<T>(string name) where T : struct, Enum
    {
        if (Options.TryGetValue(name, out var val) && val is string s)
            return StringToEnum<T>(s);
        return null;
    }

    private static string EnumToString<T>(T value) where T : Enum
    {
        return value switch
        {
            Modelling.LinearSolver e => LinearSolverMap[e],
            Modelling.HessianApproximation e => HessianApproximationMap[e],
            Modelling.MuStrategy e => MuStrategyMap[e],
            Modelling.NlpScalingMethod e => NlpScalingMethodMap[e],
            Modelling.LinearSystemScaling e => LinearSystemScalingMap[e],
            Modelling.FixedVariableTreatment e => FixedVariableTreatmentMap[e],
            Modelling.DerivativeTest e => DerivativeTestMap[e],
            _ => throw new ArgumentException($"Unknown enum type: {typeof(T)}")
        };
    }

    private static T? StringToEnum<T>(string value) where T : struct, Enum
    {
        if (typeof(T) == typeof(Modelling.LinearSolver))
        {
            var pair = LinearSolverMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(Modelling.HessianApproximation))
        {
            var pair = HessianApproximationMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(Modelling.MuStrategy))
        {
            var pair = MuStrategyMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(Modelling.NlpScalingMethod))
        {
            var pair = NlpScalingMethodMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(Modelling.LinearSystemScaling))
        {
            var pair = LinearSystemScalingMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(Modelling.FixedVariableTreatment))
        {
            var pair = FixedVariableTreatmentMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(Modelling.DerivativeTest))
        {
            var pair = DerivativeTestMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        return null;
    }
}
