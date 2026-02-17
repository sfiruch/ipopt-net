using System.Runtime.CompilerServices;

namespace IpoptNet.Modelling;

public sealed class Variable
{
    public int Index { get; internal set; } = -1;
    public double LowerBound = double.NegativeInfinity;
    public double UpperBound = double.PositiveInfinity;
    public double? Start;
    public double LowerBoundDualStart = 0;
    public double UpperBoundDualStart = 0;

    internal readonly Expr _expr;

    internal Variable()
    {
        _expr = new Expr(new VariableNode(this));
    }

    public Variable(double lowerBound, double upperBound)
    {
        LowerBound = lowerBound;
        UpperBound = upperBound;
        _expr = new Expr(new VariableNode(this));
    }

    public override bool Equals(object? obj) => ReferenceEquals(this, obj);
    public override int GetHashCode() => RuntimeHelpers.GetHashCode(this);

    // Variable OP Expr - needed to avoid ambiguity when both Variable->Expr and Expr OP Expr exist
    public static Expr operator +(Variable a, Expr b) => a._expr + b;
    public static Expr operator -(Variable a, Expr b) => a._expr - b;
    public static Expr operator *(Variable a, Expr b) => a._expr * b;
    public static Expr operator /(Variable a, Expr b) => a._expr / b;

    // double OP Variable - implicit conversion only applies to right operand in binary operators
    public static Expr operator +(double a, Variable b) => a + b._expr;
    public static Expr operator -(double a, Variable b) => a - b._expr;
    public static Expr operator *(double a, Variable b) => a * b._expr;
    public static Expr operator /(double a, Variable b) => a / b._expr;

    public static Constraint operator >=(Variable expr, double value) => expr._expr >= value;
    public static Constraint operator <=(Variable expr, double value) => expr._expr <= value;
    public static Constraint operator ==(Variable expr, double value) => expr._expr == value;
    public static Constraint operator !=(Variable expr, double value) => throw new NotSupportedException("Inequality constraints (!=) are not supported in optimization models.");

    public static Constraint operator >=(double value, Variable expr) => value >= expr._expr;
    public static Constraint operator <=(double value, Variable expr) => value <= expr._expr;
    public static Constraint operator ==(double value, Variable expr) => value == expr._expr;
    public static Constraint operator !=(double value, Variable expr) => throw new NotSupportedException("Inequality constraints (!=) are not supported in optimization models.");

    public static Constraint operator >=(Variable left, Variable right) => left._expr >= right._expr;
    public static Constraint operator <=(Variable left, Variable right) => left._expr <= right._expr;
    public static Constraint operator ==(Variable left, Variable right) => left._expr == right._expr;
    public static Constraint operator !=(Variable left, Variable right) => throw new NotSupportedException("Inequality constraints (!=) are not supported in optimization models.");

    public static Constraint operator >=(Variable left, Expr right) => left._expr >= right;
    public static Constraint operator <=(Variable left, Expr right) => left._expr <= right;
    public static Constraint operator ==(Variable left, Expr right) => left._expr == right;
    public static Constraint operator !=(Variable left, Expr right) => throw new NotSupportedException("Inequality constraints (!=) are not supported in optimization models.");

    public static Constraint operator >=(Expr left, Variable right) => left >= right._expr;
    public static Constraint operator <=(Expr left, Variable right) => left <= right._expr;
    public static Constraint operator ==(Expr left, Variable right) => left == right._expr;
    public static Constraint operator !=(Expr left, Variable right) => throw new NotSupportedException("Inequality constraints (!=) are not supported in optimization models.");
}
