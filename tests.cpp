#include <iostream>
#include <cassert>
#include <cmath>
#include <functional>
#include "autodiff.cpp"

using namespace std;

// A utility to check if two doubles are close enough
bool is_close(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) < tol;
}

// Finite Difference Gradient Checker
// Evaluates the numerical gradient of a function f with respect to a single variable at x.
double gradient_check(std::function<Var(Var)> f, double x, double eps = 1e-5) {
    // must clear the tape between evaluations to prevent memory leaks in the Wengert list
    global_tape.clear();
    double y_plus = f(Var(x + eps)).val();
    
    global_tape.clear();
    double y_minus = f(Var(x - eps)).val();
    
    global_tape.clear();
    return (y_plus - y_minus) / (2 * eps);
}

// TEST SUITE

void test_basic_operations() {
    global_tape.clear();
    Var x(2.0);
    Var y(3.0);
    Var z = x * y + sin(x) / y;
    z.backward();
    
    cout << "Basic operations compiled and ran successfully.\n";
}

void test_gradient_checking() {
    cout << "Running Gradient Checks...\n";

    // Test 1: f(x) = x^2 + 3x + 5
    auto f1 = [](Var x) { return pow(x, 2.0) + x * 3.0 + 5.0; };
    double x_val = 2.0;
    
    // Analytic
    global_tape.clear();
    Var x1(x_val);
    Var y1 = f1(x1);
    y1.backward();
    double analytic_grad = x1.grad();

    // Numerical
    double numeric_grad = gradient_check(f1, x_val);
    
    assert(is_close(analytic_grad, numeric_grad));
    cout << "  [PASS] Polynomial gradient check.\n";

    // Test 2: Composite Trig + Exp f(x) = exp(sin(x)) * tanh(x)
    auto f2 = [](Var x) { return exp(sin(x)) * tanh(x); };
    
    global_tape.clear();
    Var x2(0.5);
    Var y2 = f2(x2);
    y2.backward();
    analytic_grad = x2.grad();
    numeric_grad = gradient_check(f2, 0.5);

    assert(is_close(analytic_grad, numeric_grad));
    cout << "  [PASS] Composite Trig/Exp gradient check.\n";
}

void test_edge_cases() {
    cout << "Running Edge Case Tests...\n";

    // Edge case 1: ReLU at negative (should be 0)
    global_tape.clear();
    Var x1(-5.0);
    Var y1 = relu(x1);
    y1.backward();
    assert(is_close(x1.grad(), 0.0));
    cout << "  [PASS] ReLU negative edge case.\n";

    // Edge case 2: Variable used multiple times (graph branching)
    // f(x) = x * x * x
    global_tape.clear();
    Var x2(3.0);
    Var y2 = x2 * x2 * x2; 
    y2.backward();
    // f'(x) = 3x^2. If x=3, f'(3) = 27
    assert(is_close(x2.grad(), 27.0)); 
    cout << "  [PASS] Graph branching (x used multiple times) correctly accumulates gradients.\n";
}

int main() {
    cout << "=== Autodiff Test Suite ===\n";
    test_basic_operations();
    test_gradient_checking();
    test_edge_cases();
    cout << "All tests passed successfully!\n";
    return 0;
}