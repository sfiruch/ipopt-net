namespace IpoptNet.Modelling;

public sealed class Constraint
{
    public Expr Expression { get; }
    public double LowerBound { get; }
    public double UpperBound { get; }
    public double DualStart { get; set; } = 0;

    public Constraint(Expr expression, double lowerBound, double upperBound)
    {
        Expression = expression;
        LowerBound = lowerBound;
        UpperBound = upperBound;
    }

    public static Constraint LessThanOrEqual(Expr expr, double upper) =>
        new(expr, double.NegativeInfinity, upper);

    public static Constraint GreaterThanOrEqual(Expr expr, double lower) =>
        new(expr, lower, double.PositiveInfinity);

    public static Constraint Equal(Expr expr, double value) =>
        new(expr, value, value);

    public static Constraint Between(Expr expr, double lower, double upper) =>
        new(expr, lower, upper);
}
