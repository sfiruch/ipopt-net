using IpoptNet.Modelling;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace IpoptNet.Tests;

[TestClass]
public class OperatorSymmetryTests
{
    [TestMethod]
    public void Expr_Plus_Variable_WorksSymmetrically()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr expr = x + 5;
        var result1 = expr + y; // Expr + Variable
        var result2 = y + expr; // Variable + Expr

        double[] point = [2, 3];
        Assert.AreEqual(result1.Evaluate(point), result2.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void Expr_Minus_Variable_WorksSymmetrically()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr expr = x + 5;
        var result1 = expr - y; // Expr - Variable
        var result2 = -(y - expr); // Negation of (Variable - Expr)

        double[] point = [2, 3];
        Assert.AreEqual(result1.Evaluate(point), result2.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void Expr_Multiply_Variable_WorksSymmetrically()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr expr = x + 5;
        var result1 = expr * y; // Expr * Variable
        var result2 = y * expr; // Variable * Expr

        double[] point = [2, 3];
        Assert.AreEqual(result1.Evaluate(point), result2.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void Expr_Divide_Variable_Works()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr expr = x + 5;
        var result = expr / y; // Expr / Variable

        double[] point = [2, 3];
        double expected = (2.0 + 5.0) / 3.0; // 7/3
        Assert.AreEqual(expected, result.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void Variable_Divide_Expr_Works()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr expr = y + 5;
        var result = x / expr; // Variable / Expr

        double[] point = [2, 3];
        double expected = 2.0 / (3.0 + 5.0); // 2/8
        Assert.AreEqual(expected, result.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void Expr_ComparisonOperators_Variable_Work()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        Expr expr = x + 5;
        
        // Test all comparison operators
        var c1 = expr >= y;
        var c2 = expr <= y;
        var c3 = expr == y;

        Assert.IsNotNull(c1);
        Assert.IsNotNull(c2);
        Assert.IsNotNull(c3);
    }

    [TestMethod]
    public void Variable_Purpose_HasBoth_VarNode_And_Expr()
    {
        // This test documents why Variable needs both _varNode and _expr:
        // - _varNode is the actual node representation used when building expression trees
        // - _expr is a cached Expr wrapper for efficient operator overloading
        
        var model = new Model();
        var x = model.AddVariable();

        // _expr is used for operators - avoids creating new Expr wrapper each time
        var expr1 = x + 5; // Uses x._expr internally
        var expr2 = x * 2; // Uses x._expr internally
        
        // Both expressions should reference the same underlying VariableNode
        // (though wrapped in different expression trees)
        var vars1 = new HashSet<Variable>();
        var vars2 = new HashSet<Variable>();
        expr1.CollectVariables(vars1);
        expr2.CollectVariables(vars2);
        
        Assert.IsTrue(vars1.Contains(x));
        Assert.IsTrue(vars2.Contains(x));
        Assert.AreSame(vars1.Single(), vars2.Single(), "Both expressions should reference the same Variable instance");
    }

    [TestMethod]
    public void ImplicitConversion_Variable_To_Expr_Works()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Implicit conversion should work
        Expr expr = x;
        
        Assert.IsNotNull(expr);
        double[] point = [5];
        Assert.AreEqual(5, expr.Evaluate(point));
    }

    [TestMethod]
    public void ImplicitConversion_EnablesSymmetricOperators()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // These should all work due to implicit conversion Variable -> Expr
        Expr expr = x + 3;
        var r1 = expr + y; // Expr + Variable (implicit conversion)
        var r2 = expr - y; // Expr - Variable (implicit conversion)
        var r3 = expr * y; // Expr * Variable (implicit conversion)
        var r4 = expr / y; // Expr / Variable (implicit conversion)

        // Verify they produce correct results
        double[] point = [2, 3];
        Assert.AreEqual((2 + 3) + 3, r1.Evaluate(point), 1e-10);
        Assert.AreEqual((2 + 3) - 3, r2.Evaluate(point), 1e-10);
        Assert.AreEqual((2 + 3) * 3, r3.Evaluate(point), 1e-10);
        Assert.AreEqual((2 + 3) / 3.0, r4.Evaluate(point), 1e-10);
        
        // Now test the reverse: Variable OP Expr
        var r5 = y + expr; // Variable + Expr (implicit conversion)
        var r6 = y - expr; // Variable - Expr (implicit conversion)
        var r7 = y * expr; // Variable * Expr (implicit conversion)
        var r8 = y / expr; // Variable / Expr (implicit conversion)

        Assert.AreEqual(3 + (2 + 3), r5.Evaluate(point), 1e-10);
        Assert.AreEqual(3 - (2 + 3), r6.Evaluate(point), 1e-10);
        Assert.AreEqual(3 * (2 + 3), r7.Evaluate(point), 1e-10);
        Assert.AreEqual(3 / (2.0 + 3.0), r8.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void Variable_OP_Variable_AlsoUsesImplicitConversion()
    {
        var model = new Model();
        var x = model.AddVariable();
        var y = model.AddVariable();

        // Even Variable OP Variable could use implicit conversion
        var r1 = x + y;
        var r2 = x - y;
        var r3 = x * y;
        var r4 = x / y;

        double[] point = [6, 3];
        Assert.AreEqual(6 + 3, r1.Evaluate(point), 1e-10);
        Assert.AreEqual(6 - 3, r2.Evaluate(point), 1e-10);
        Assert.AreEqual(6 * 3, r3.Evaluate(point), 1e-10);
        Assert.AreEqual(6 / 3.0, r4.Evaluate(point), 1e-10);
    }

    [TestMethod]
    public void Variable_OP_Double_AlsoUsesImplicitConversion()
    {
        var model = new Model();
        var x = model.AddVariable();

        // Variable OP double could also use implicit conversion
        var r1 = x + 5.0;
        var r2 = x - 3.0;
        var r3 = x * 2.0;
        var r4 = x / 4.0;

        double[] point = [8];
        Assert.AreEqual(8 + 5, r1.Evaluate(point), 1e-10);
        Assert.AreEqual(8 - 3, r2.Evaluate(point), 1e-10);
        Assert.AreEqual(8 * 2, r3.Evaluate(point), 1e-10);
        Assert.AreEqual(8 / 4.0, r4.Evaluate(point), 1e-10);
    }
}
