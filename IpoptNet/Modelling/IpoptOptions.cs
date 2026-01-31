namespace IpoptNet.Modelling;

/// <summary>
/// Documents the IPOPT string value for an enum. For documentation only - actual mappings are in IpoptOptions dictionaries.
/// </summary>
[AttributeUsage(AttributeTargets.Field)]
internal sealed class IpoptValueAttribute : Attribute
{
    public string Value { get; }
    public IpoptValueAttribute(string value) => Value = value;
}

/// <summary>
/// Available linear solvers for IPOPT.
/// </summary>
public enum LinearSolver
{
    /// <summary>MUltifrontal Massively Parallel Sparse direct solver (default).</summary>
    [IpoptValue("mumps")]
    Mumps,
    /// <summary>MA27 from Harwell Subroutines Library.</summary>
    [IpoptValue("ma27")]
    Ma27,
    /// <summary>MA57 from Harwell Subroutines Library.</summary>
    [IpoptValue("ma57")]
    Ma57,
    /// <summary>MA77 from Harwell Subroutines Library.</summary>
    [IpoptValue("ma77")]
    Ma77,
    /// <summary>MA86 from Harwell Subroutines Library.</summary>
    [IpoptValue("ma86")]
    Ma86,
    /// <summary>MA97 from Harwell Subroutines Library.</summary>
    [IpoptValue("ma97")]
    Ma97,
    /// <summary>Pardiso from Intel Math Kernel Library (MKL).</summary>
    [IpoptValue("pardisomkl")]
    PardisoMkl,
    /// <summary>Pardiso from pardiso-project.org.</summary>
    [IpoptValue("pardiso")]
    PardisoProject,
    /// <summary>Watson Sparse Matrix Package.</summary>
    [IpoptValue("wsmp")]
    Wsmp,
    /// <summary>Sparse Parallel Robust Algorithms Library.</summary>
    [IpoptValue("spral")]
    Spral,
    /// <summary>Custom linear solver.</summary>
    [IpoptValue("custom")]
    Custom
}

/// <summary>
/// Hessian approximation methods.
/// </summary>
public enum HessianApproximation
{
    /// <summary>Use exact Hessian (default).</summary>
    [IpoptValue("exact")]
    Exact,
    /// <summary>Use limited-memory quasi-Newton approximation.</summary>
    [IpoptValue("limited-memory")]
    LimitedMemory
}

/// <summary>
/// Barrier parameter update strategies.
/// </summary>
public enum MuStrategy
{
    /// <summary>Monotone decrease of barrier parameter.</summary>
    [IpoptValue("monotone")]
    Monotone,
    /// <summary>Adaptive update of barrier parameter.</summary>
    [IpoptValue("adaptive")]
    Adaptive
}

/// <summary>
/// NLP scaling methods.
/// </summary>
public enum NlpScalingMethod
{
    /// <summary>No scaling.</summary>
    [IpoptValue("none")]
    None,
    /// <summary>User-provided scaling.</summary>
    [IpoptValue("user-scaling")]
    UserScaling,
    /// <summary>Gradient-based scaling.</summary>
    [IpoptValue("gradient-based")]
    GradientBased,
    /// <summary>Equilibration-based scaling.</summary>
    [IpoptValue("equilibration-based")]
    EquilibrationBased
}

/// <summary>
/// Linear system scaling methods.
/// </summary>
public enum LinearSystemScaling
{
    /// <summary>No scaling.</summary>
    [IpoptValue("none")]
    None,
    /// <summary>MC19 scaling from HSL.</summary>
    [IpoptValue("mc19")]
    Mc19,
    /// <summary>Slack-based scaling.</summary>
    [IpoptValue("slack-based")]
    SlackBased
}

/// <summary>
/// Treatment of fixed variables.
/// </summary>
public enum FixedVariableTreatment
{
    /// <summary>Make fixed variables parameters.</summary>
    [IpoptValue("make_parameter")]
    MakeParameter,
    /// <summary>Add equality constraints for fixed variables.</summary>
    [IpoptValue("make_constraint")]
    MakeConstraint,
    /// <summary>Relax bounds slightly.</summary>
    [IpoptValue("relax_bounds")]
    RelaxBounds
}

/// <summary>
/// Derivative test options.
/// </summary>
public enum DerivativeTest
{
    /// <summary>No derivative test.</summary>
    [IpoptValue("none")]
    None,
    /// <summary>Test first derivatives only.</summary>
    [IpoptValue("first-order")]
    FirstOrder,
    /// <summary>Test second derivatives only.</summary>
    [IpoptValue("second-order")]
    SecondOrder,
    /// <summary>Test only at starting point.</summary>
    [IpoptValue("only-second-order")]
    OnlySecondOrder
}

/// <summary>
/// Options for IPOPT solver configuration.
/// </summary>
public sealed class IpoptOptions
{
    internal readonly Dictionary<string, object> Options = new();

    // AOT-friendly mappings (single source of truth)
    private static readonly Dictionary<LinearSolver, string> LinearSolverMap = new()
    {
        [LinearSolver.Mumps] = "mumps",
        [LinearSolver.Ma27] = "ma27",
        [LinearSolver.Ma57] = "ma57",
        [LinearSolver.Ma77] = "ma77",
        [LinearSolver.Ma86] = "ma86",
        [LinearSolver.Ma97] = "ma97",
        [LinearSolver.PardisoMkl] = "pardisomkl",
        [LinearSolver.PardisoProject] = "pardiso",
        [LinearSolver.Wsmp] = "wsmp",
        [LinearSolver.Spral] = "spral",
        [LinearSolver.Custom] = "custom"
    };

    private static readonly Dictionary<HessianApproximation, string> HessianApproximationMap = new()
    {
        [HessianApproximation.Exact] = "exact",
        [HessianApproximation.LimitedMemory] = "limited-memory"
    };

    private static readonly Dictionary<MuStrategy, string> MuStrategyMap = new()
    {
        [MuStrategy.Monotone] = "monotone",
        [MuStrategy.Adaptive] = "adaptive"
    };

    private static readonly Dictionary<NlpScalingMethod, string> NlpScalingMethodMap = new()
    {
        [NlpScalingMethod.None] = "none",
        [NlpScalingMethod.UserScaling] = "user-scaling",
        [NlpScalingMethod.GradientBased] = "gradient-based",
        [NlpScalingMethod.EquilibrationBased] = "equilibration-based"
    };

    private static readonly Dictionary<LinearSystemScaling, string> LinearSystemScalingMap = new()
    {
        [LinearSystemScaling.None] = "none",
        [LinearSystemScaling.Mc19] = "mc19",
        [LinearSystemScaling.SlackBased] = "slack-based"
    };

    private static readonly Dictionary<FixedVariableTreatment, string> FixedVariableTreatmentMap = new()
    {
        [FixedVariableTreatment.MakeParameter] = "make_parameter",
        [FixedVariableTreatment.MakeConstraint] = "make_constraint",
        [FixedVariableTreatment.RelaxBounds] = "relax_bounds"
    };

    private static readonly Dictionary<DerivativeTest, string> DerivativeTestMap = new()
    {
        [DerivativeTest.None] = "none",
        [DerivativeTest.FirstOrder] = "first-order",
        [DerivativeTest.SecondOrder] = "second-order",
        [DerivativeTest.OnlySecondOrder] = "only-second-order"
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
    public LinearSolver? LinearSolverOption
    {
        get => GetEnum<LinearSolver>("linear_solver");
        set => SetEnum("linear_solver", value);
    }

    public HessianApproximation? HessianApproximationOption
    {
        get => GetEnum<HessianApproximation>("hessian_approximation");
        set => SetEnum("hessian_approximation", value);
    }

    public MuStrategy? MuStrategyOption
    {
        get => GetEnum<MuStrategy>("mu_strategy");
        set => SetEnum("mu_strategy", value);
    }

    public NlpScalingMethod? NlpScalingMethodOption
    {
        get => GetEnum<NlpScalingMethod>("nlp_scaling_method");
        set => SetEnum("nlp_scaling_method", value);
    }

    public LinearSystemScaling? LinearSystemScalingOption
    {
        get => GetEnum<LinearSystemScaling>("linear_system_scaling");
        set => SetEnum("linear_system_scaling", value);
    }

    public FixedVariableTreatment? FixedVariableTreatmentOption
    {
        get => GetEnum<FixedVariableTreatment>("fixed_variable_treatment");
        set => SetEnum("fixed_variable_treatment", value);
    }

    public DerivativeTest? DerivativeTestOption
    {
        get => GetEnum<DerivativeTest>("derivative_test");
        set => SetEnum("derivative_test", value);
    }

    // Constraint/NLP options
    public double? ConstraintViolationTolerance { get => GetDouble("constr_viol_tol"); set => SetDouble("constr_viol_tol", value); }
    public double? DualInfeasibilityTolerance { get => GetDouble("dual_inf_tol"); set => SetDouble("dual_inf_tol", value); }
    public double? ComplementarityTolerance { get => GetDouble("compl_inf_tol"); set => SetDouble("compl_inf_tol", value); }

    // Initialization options
    public double? BoundPush { get => GetDouble("bound_push"); set => SetDouble("bound_push", value); }
    public double? BoundFraction { get => GetDouble("bound_frac"); set => SetDouble("bound_frac", value); }
    public bool? WarmStartInitPoint { get => GetBool("warm_start_init_point"); set => SetBool("warm_start_init_point", value); }

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
            LinearSolver e => LinearSolverMap[e],
            HessianApproximation e => HessianApproximationMap[e],
            MuStrategy e => MuStrategyMap[e],
            NlpScalingMethod e => NlpScalingMethodMap[e],
            LinearSystemScaling e => LinearSystemScalingMap[e],
            FixedVariableTreatment e => FixedVariableTreatmentMap[e],
            DerivativeTest e => DerivativeTestMap[e],
            _ => throw new ArgumentException($"Unknown enum type: {typeof(T)}")
        };
    }

    private static T? StringToEnum<T>(string value) where T : struct, Enum
    {
        if (typeof(T) == typeof(LinearSolver))
        {
            var pair = LinearSolverMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(HessianApproximation))
        {
            var pair = HessianApproximationMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(MuStrategy))
        {
            var pair = MuStrategyMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(NlpScalingMethod))
        {
            var pair = NlpScalingMethodMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(LinearSystemScaling))
        {
            var pair = LinearSystemScalingMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(FixedVariableTreatment))
        {
            var pair = FixedVariableTreatmentMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        if (typeof(T) == typeof(DerivativeTest))
        {
            var pair = DerivativeTestMap.FirstOrDefault(kvp => kvp.Value.Equals(value, StringComparison.OrdinalIgnoreCase));
            return pair.Key != default ? (T)(object)pair.Key : null;
        }
        return null;
    }
}
