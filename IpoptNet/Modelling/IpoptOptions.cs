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
    public double? ScalingMaximum { get => GetDouble("s_max"); set => SetDouble("s_max", value); }
    public int? MaxIterations { get => GetInt("max_iter"); set => SetInt("max_iter", value); }
    public double? MaxWallTime { get => GetDouble("max_wall_time"); set => SetDouble("max_wall_time", value); }
    public double? MaxCpuTime { get => GetDouble("max_cpu_time"); set => SetDouble("max_cpu_time", value); }
    public double? DualInfeasibilityTolerance { get => GetDouble("dual_inf_tol"); set => SetDouble("dual_inf_tol", value); }
    public double? ConstraintViolationTolerance { get => GetDouble("constr_viol_tol"); set => SetDouble("constr_viol_tol", value); }
    public double? ComplementarityTolerance { get => GetDouble("compl_inf_tol"); set => SetDouble("compl_inf_tol", value); }
    public double? AcceptableTolerance { get => GetDouble("acceptable_tol"); set => SetDouble("acceptable_tol", value); }
    public int? AcceptableIterations { get => GetInt("acceptable_iter"); set => SetInt("acceptable_iter", value); }
    public double? AcceptableDualInfeasibilityTolerance { get => GetDouble("acceptable_dual_inf_tol"); set => SetDouble("acceptable_dual_inf_tol", value); }
    public double? AcceptableConstraintViolationTolerance { get => GetDouble("acceptable_constr_viol_tol"); set => SetDouble("acceptable_constr_viol_tol", value); }
    public double? AcceptableComplementarityTolerance { get => GetDouble("acceptable_compl_inf_tol"); set => SetDouble("acceptable_compl_inf_tol", value); }
    public double? AcceptableObjectiveChangeTolerance { get => GetDouble("acceptable_obj_change_tol"); set => SetDouble("acceptable_obj_change_tol", value); }
    public double? DivergingIteratesTolerance { get => GetDouble("diverging_iterates_tol"); set => SetDouble("diverging_iterates_tol", value); }
    public double? MuTarget { get => GetDouble("mu_target"); set => SetDouble("mu_target", value); }

    // Output options
    public int? PrintLevel { get => GetInt("print_level"); set => SetInt("print_level", value); }
    public string? OutputFile { get => GetString("output_file"); set => SetString("output_file", value); }
    public int? FilePrintLevel { get => GetInt("file_print_level"); set => SetInt("file_print_level", value); }
    public bool? FileAppend { get => GetBool("file_append"); set => SetBool("file_append", value); }
    public bool? PrintUserOptions { get => GetBool("print_user_options"); set => SetBool("print_user_options", value); }
    public bool? PrintOptionsDocumentation { get => GetBool("print_options_documentation"); set => SetBool("print_options_documentation", value); }
    public bool? PrintTimingStatistics { get => GetBool("print_timing_statistics"); set => SetBool("print_timing_statistics", value); }
    public bool? PrintAdvancedOptions { get => GetBool("print_advanced_options"); set => SetBool("print_advanced_options", value); }
    public bool? PrintInfoString { get => GetBool("print_info_string"); set => SetBool("print_info_string", value); }
    public int? PrintFrequencyIter { get => GetInt("print_frequency_iter"); set => SetInt("print_frequency_iter", value); }
    public double? PrintFrequencyTime { get => GetDouble("print_frequency_time"); set => SetDouble("print_frequency_time", value); }

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

    // Barrier parameter options
    public double? MuInit { get => GetDouble("mu_init"); set => SetDouble("mu_init", value); }
    public double? MuMin { get => GetDouble("mu_min"); set => SetDouble("mu_min", value); }
    public double? MuMax { get => GetDouble("mu_max"); set => SetDouble("mu_max", value); }
    public double? MuMaxFact { get => GetDouble("mu_max_fact"); set => SetDouble("mu_max_fact", value); }
    public double? MuLinearDecreaseFactor { get => GetDouble("mu_linear_decrease_factor"); set => SetDouble("mu_linear_decrease_factor", value); }
    public double? MuSuperlinearDecreasePower { get => GetDouble("mu_superlinear_decrease_power"); set => SetDouble("mu_superlinear_decrease_power", value); }
    public double? BarrierToleranceFactor { get => GetDouble("barrier_tol_factor"); set => SetDouble("barrier_tol_factor", value); }
    public double? TauMin { get => GetDouble("tau_min"); set => SetDouble("tau_min", value); }

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
    public int? DerivativeTestFirstIndex { get => GetInt("derivative_test_first_index"); set => SetInt("derivative_test_first_index", value); }
    public bool? DerivativeTestPrintAll { get => GetBool("derivative_test_print_all"); set => SetBool("derivative_test_print_all", value); }

    // NLP options
    public double? NlpLowerBoundInf { get => GetDouble("nlp_lower_bound_inf"); set => SetDouble("nlp_lower_bound_inf", value); }
    public double? NlpUpperBoundInf { get => GetDouble("nlp_upper_bound_inf"); set => SetDouble("nlp_upper_bound_inf", value); }
    public double? BoundRelaxFactor { get => GetDouble("bound_relax_factor"); set => SetDouble("bound_relax_factor", value); }
    public bool? HonorOriginalBounds { get => GetBool("honor_original_bounds"); set => SetBool("honor_original_bounds", value); }
    public bool? CheckDerivativesForNanInf { get => GetBool("check_derivatives_for_naninf"); set => SetBool("check_derivatives_for_naninf", value); }

    // NLP Scaling options
    public double? ObjectiveScalingFactor { get => GetDouble("obj_scaling_factor"); set => SetDouble("obj_scaling_factor", value); }
    public double? NlpScalingMaxGradient { get => GetDouble("nlp_scaling_max_gradient"); set => SetDouble("nlp_scaling_max_gradient", value); }
    public double? NlpScalingMinValue { get => GetDouble("nlp_scaling_min_value"); set => SetDouble("nlp_scaling_min_value", value); }

    // Initialization options
    public double? BoundPush { get => GetDouble("bound_push"); set => SetDouble("bound_push", value); }
    public double? BoundFraction { get => GetDouble("bound_frac"); set => SetDouble("bound_frac", value); }
    public double? SlackBoundPush { get => GetDouble("slack_bound_push"); set => SetDouble("slack_bound_push", value); }
    public double? SlackBoundFraction { get => GetDouble("slack_bound_frac"); set => SetDouble("slack_bound_frac", value); }
    public double? ConstraintMultiplierInitMax { get => GetDouble("constr_mult_init_max"); set => SetDouble("constr_mult_init_max", value); }
    public double? BoundMultiplierInitValue { get => GetDouble("bound_mult_init_val"); set => SetDouble("bound_mult_init_val", value); }

    // Warm start options
    public bool? WarmStartInitPoint { get => GetBool("warm_start_init_point"); set => SetBool("warm_start_init_point", value); }
    public bool? WarmStartSameStructure { get => GetBool("warm_start_same_structure"); set => SetBool("warm_start_same_structure", value); }
    public double? WarmStartBoundPush { get => GetDouble("warm_start_bound_push"); set => SetDouble("warm_start_bound_push", value); }
    public double? WarmStartBoundFrac { get => GetDouble("warm_start_bound_frac"); set => SetDouble("warm_start_bound_frac", value); }
    public double? WarmStartMultBoundPush { get => GetDouble("warm_start_mult_bound_push"); set => SetDouble("warm_start_mult_bound_push", value); }
    public double? WarmStartSlackBoundPush { get => GetDouble("warm_start_slack_bound_push"); set => SetDouble("warm_start_slack_bound_push", value); }
    public double? WarmStartSlackBoundFrac { get => GetDouble("warm_start_slack_bound_frac"); set => SetDouble("warm_start_slack_bound_frac", value); }
    public double? WarmStartMultiplierInitMax { get => GetDouble("warm_start_mult_init_max"); set => SetDouble("warm_start_mult_init_max", value); }

    // Line search options
    public double? AlphaReductionFactor { get => GetDouble("alpha_red_factor"); set => SetDouble("alpha_red_factor", value); }
    public bool? AcceptEveryTrialStep { get => GetBool("accept_every_trial_step"); set => SetBool("accept_every_trial_step", value); }
    public int? AcceptAfterMaxSteps { get => GetInt("accept_after_max_steps"); set => SetInt("accept_after_max_steps", value); }
    public double? TinyStepTolerance { get => GetDouble("tiny_step_tol"); set => SetDouble("tiny_step_tol", value); }
    public int? MaxSecondOrderCorrection { get => GetInt("max_soc"); set => SetInt("max_soc", value); }
    public double? ObjectiveMaxIncrease { get => GetDouble("obj_max_inc"); set => SetDouble("obj_max_inc", value); }

    // Step calculation options
    public bool? MehrotraAlgorithm { get => GetBool("mehrotra_algorithm"); set => SetBool("mehrotra_algorithm", value); }
    public int? MinRefinementSteps { get => GetInt("min_refinement_steps"); set => SetInt("min_refinement_steps", value); }
    public int? MaxRefinementSteps { get => GetInt("max_refinement_steps"); set => SetInt("max_refinement_steps", value); }
    public double? MaxHessianPerturbation { get => GetDouble("max_hessian_perturbation"); set => SetDouble("max_hessian_perturbation", value); }
    public double? MinHessianPerturbation { get => GetDouble("min_hessian_perturbation"); set => SetDouble("min_hessian_perturbation", value); }
    public double? FirstHessianPerturbation { get => GetDouble("first_hessian_perturbation"); set => SetDouble("first_hessian_perturbation", value); }
    public double? JacobianRegularizationValue { get => GetDouble("jacobian_regularization_value"); set => SetDouble("jacobian_regularization_value", value); }

    // Restoration phase options
    public bool? ExpectInfeasibleProblem { get => GetBool("expect_infeasible_problem"); set => SetBool("expect_infeasible_problem", value); }
    public double? ExpectInfeasibleProblemConstraintTolerance { get => GetDouble("expect_infeasible_problem_ctol"); set => SetDouble("expect_infeasible_problem_ctol", value); }
    public double? ExpectInfeasibleProblemMultiplierTolerance { get => GetDouble("expect_infeasible_problem_ytol"); set => SetDouble("expect_infeasible_problem_ytol", value); }
    public bool? StartWithRestoration { get => GetBool("start_with_resto"); set => SetBool("start_with_resto", value); }
    public double? RequiredInfeasibilityReduction { get => GetDouble("required_infeasibility_reduction"); set => SetDouble("required_infeasibility_reduction", value); }
    public int? MaxRestorationIterations { get => GetInt("max_resto_iter"); set => SetInt("max_resto_iter", value); }

    // Limited memory options
    public int? LimitedMemoryMaxHistory { get => GetInt("limited_memory_max_history"); set => SetInt("limited_memory_max_history", value); }
    public double? LimitedMemoryInitValue { get => GetDouble("limited_memory_init_val"); set => SetDouble("limited_memory_init_val", value); }
    public double? LimitedMemoryInitValueMax { get => GetDouble("limited_memory_init_val_max"); set => SetDouble("limited_memory_init_val_max", value); }
    public double? LimitedMemoryInitValueMin { get => GetDouble("limited_memory_init_val_min"); set => SetDouble("limited_memory_init_val_min", value); }

    // Linear solver specific options - MA27
    public int? Ma27PrintLevel { get => GetInt("ma27_print_level"); set => SetInt("ma27_print_level", value); }
    public double? Ma27PivotTolerance { get => GetDouble("ma27_pivtol"); set => SetDouble("ma27_pivtol", value); }
    public double? Ma27PivotToleranceMax { get => GetDouble("ma27_pivtolmax"); set => SetDouble("ma27_pivtolmax", value); }
    public double? Ma27IntegerWorkspaceMemoryFactor { get => GetDouble("ma27_liw_init_factor"); set => SetDouble("ma27_liw_init_factor", value); }
    public double? Ma27RealWorkspaceMemoryFactor { get => GetDouble("ma27_la_init_factor"); set => SetDouble("ma27_la_init_factor", value); }
    public double? Ma27MemoryIncrementFactor { get => GetDouble("ma27_meminc_factor"); set => SetDouble("ma27_meminc_factor", value); }

    // Linear solver specific options - MA57
    public int? Ma57PrintLevel { get => GetInt("ma57_print_level"); set => SetInt("ma57_print_level", value); }
    public double? Ma57PivotTolerance { get => GetDouble("ma57_pivtol"); set => SetDouble("ma57_pivtol", value); }
    public double? Ma57PivotToleranceMax { get => GetDouble("ma57_pivtolmax"); set => SetDouble("ma57_pivtolmax", value); }
    public bool? Ma57AutomaticScaling { get => GetBool("ma57_automatic_scaling"); set => SetBool("ma57_automatic_scaling", value); }
    public double? Ma57PreAlloc { get => GetDouble("ma57_pre_alloc"); set => SetDouble("ma57_pre_alloc", value); }
    public int? Ma57PivotOrder { get => GetInt("ma57_pivot_order"); set => SetInt("ma57_pivot_order", value); }

    // Linear solver specific options - Mumps
    public int? MumpsPrintLevel { get => GetInt("mumps_print_level"); set => SetInt("mumps_print_level", value); }
    public double? MumpsPivotTolerance { get => GetDouble("mumps_pivtol"); set => SetDouble("mumps_pivtol", value); }
    public double? MumpsPivotToleranceMax { get => GetDouble("mumps_pivtolmax"); set => SetDouble("mumps_pivtolmax", value); }
    public int? MumpsMemoryPercent { get => GetInt("mumps_mem_percent"); set => SetInt("mumps_mem_percent", value); }
    public int? MumpsPermutingScaling { get => GetInt("mumps_permuting_scaling"); set => SetInt("mumps_permuting_scaling", value); }

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
