#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>

// Represents a node in the computation graph
struct Node {
    double value;
    double adjoint; // The accumulated gradient
    std::vector<size_t> parents; // Indices of parent nodes
    std::vector<double> local_grads; // Partial derivatives with respect to parents
    std::string op_name; // Tracks the operation for visualization

    Node(double v, std::string op = "Leaf") : value(v), adjoint(0.0), op_name(op) {}
};

// The global tape (Wengert List)
struct Tape {
    std::vector<Node> nodes;

    size_t push(double value, const std::vector<size_t>& parents, const std::vector<double>& local_grads, std::string op_name = "Op") {
        nodes.push_back(Node(value, op_name));
        nodes.back().parents = parents;
        nodes.back().local_grads = local_grads;
        return nodes.size() - 1;
    }

    void clear() {
        nodes.clear();
    }

    // Function to generate Graphviz DOT format
    void export_graphviz(const std::string& filename) {
        std::ofstream out(filename);
        out << "digraph ComputationGraph {\n";
        out << "  rankdir=LR;\n"; // Draw graph Left to Right
        out << "  node [shape=record, style=filled, fontname=\"Helvetica\"];\n";

        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& n = nodes[i];
            
            // Color code: Green for inputs, Red for final output, Blue for intermediate
            std::string color = n.parents.empty() ? "\"#d4edda\"" : "\"#cce5ff\"";
            if (i == nodes.size() - 1) color = "\"#f8d7da\""; 

            // Create the node box showing ID, Operation, Value, and Gradient
            out << "  node" << i << " [label=\"{ <id> ID: " << i 
                << " | <op> Op: " << n.op_name
                << " | <val> Val: " << n.value 
                << " | <grad> Grad: " << n.adjoint 
                << " }\", fillcolor=" << color << "];\n";

            // Draw edges from parents to this node, labeled with the local gradient
            for (size_t j = 0; j < n.parents.size(); ++j) {
                out << "  node" << n.parents[j] << " -> node" << i 
                    << " [label=\" ∂=" << n.local_grads[j] << "\", fontname=\"Helvetica\", fontsize=10];\n";
            }
        }
        out << "}\n";
        std::cout << "Graph exported to " << filename << "\n";
    }
};

// Inline global tape so it can be included in multiple files (C++17 feature)
inline Tape global_tape;

// Variable wrapper class
class Var {
public:
    size_t idx;

    // Create a new leaf variable
    Var(double v) {
        idx = global_tape.push(v, {}, {}, "Input");
    }

    // Internal constructor for intermediate nodes
    Var(size_t i) : idx(i) {}

    double val() const { return global_tape.nodes[idx].value; }
    double grad() const { return global_tape.nodes[idx].adjoint; }

    // Reverse-mode differentiation
    void backward() {
        // Zero out all previous adjoints
        for (auto& n : global_tape.nodes) {
            n.adjoint = 0.0;
        }
        
        // The seed for the backward pass
        global_tape.nodes[idx].adjoint = 1.0;

        // Traverse backwards
        for (int i = global_tape.nodes.size() - 1; i >= 0; --i) {
            Node& curr = global_tape.nodes[i];
            for (size_t j = 0; j < curr.parents.size(); ++j) {
                global_tape.nodes[curr.parents[j]].adjoint += curr.adjoint * curr.local_grads[j];
            }
        }
    }
};

// ==========================================
// 15+ OPERATIONS (Operator Overloads & Math)
// ==========================================

// 1. Addition
inline Var operator+(const Var& a, const Var& b) {
    double val = a.val() + b.val();
    return Var(global_tape.push(val, {a.idx, b.idx}, {1.0, 1.0}, "Add (+)"));
}
inline Var operator+(const Var& a, double b) { return a + Var(b); }
inline Var operator+(double a, const Var& b) { return Var(a) + b; }

// 2. Subtraction
inline Var operator-(const Var& a, const Var& b) {
    double val = a.val() - b.val();
    return Var(global_tape.push(val, {a.idx, b.idx}, {1.0, -1.0}, "Subtract (-)"));
}
inline Var operator-(const Var& a, double b) { return a - Var(b); }
inline Var operator-(double a, const Var& b) { return Var(a) - b; }

// 3. Multiplication
inline Var operator*(const Var& a, const Var& b) {
    double val = a.val() * b.val();
    return Var(global_tape.push(val, {a.idx, b.idx}, {b.val(), a.val()}, "Multiply (*)"));
}
inline Var operator*(const Var& a, double b) { return a * Var(b); }
inline Var operator*(double a, const Var& b) { return Var(a) * b; }

// 4. Division
inline Var operator/(const Var& a, const Var& b) {
    double val = a.val() / b.val();
    double grad_a = 1.0 / b.val();
    double grad_b = -a.val() / (b.val() * b.val());
    return Var(global_tape.push(val, {a.idx, b.idx}, {grad_a, grad_b}, "Divide (/)"));
}

// 5. Unary Negation
inline Var operator-(const Var& a) {
    return Var(global_tape.push(-a.val(), {a.idx}, {-1.0}, "Negate (-)"));
}

// 6. Sin
inline Var sin(const Var& a) {
    return Var(global_tape.push(std::sin(a.val()), {a.idx}, {std::cos(a.val())}, "Sin"));
}

// 7. Cos
inline Var cos(const Var& a) {
    return Var(global_tape.push(std::cos(a.val()), {a.idx}, {-std::sin(a.val())}, "Cos"));
}

// 8. Tan
inline Var tan(const Var& a) {
    double val = std::tan(a.val());
    double c = std::cos(a.val());
    return Var(global_tape.push(val, {a.idx}, {1.0 / (c * c)}, "Tan"));
}

// 9. Exp
inline Var exp(const Var& a) {
    double val = std::exp(a.val());
    return Var(global_tape.push(val, {a.idx}, {val}, "Exp"));
}

// 10. Log (Natural Logarithm)
inline Var log(const Var& a) {
    return Var(global_tape.push(std::log(a.val()), {a.idx}, {1.0 / a.val()}, "Log"));
}

// 11. Power (Var ^ double)
inline Var pow(const Var& a, double p) {
    double val = std::pow(a.val(), p);
    return Var(global_tape.push(val, {a.idx}, {p * std::pow(a.val(), p - 1.0)}, "Pow (const)"));
}

// 12. Power (Var ^ Var)
inline Var pow(const Var& a, const Var& b) {
    double val = std::pow(a.val(), b.val());
    double grad_a = b.val() * std::pow(a.val(), b.val() - 1.0);
    double grad_b = val * std::log(a.val());
    return Var(global_tape.push(val, {a.idx, b.idx}, {grad_a, grad_b}, "Pow"));
}

// 13. Sqrt
inline Var sqrt(const Var& a) {
    double val = std::sqrt(a.val());
    return Var(global_tape.push(val, {a.idx}, {0.5 / val}, "Sqrt"));
}

// 14. ReLU (Rectified Linear Unit)
inline Var relu(const Var& a) {
    double val = std::max(0.0, a.val());
    double grad = a.val() > 0.0 ? 1.0 : 0.0;
    return Var(global_tape.push(val, {a.idx}, {grad}, "ReLU"));
}

// 15. Sigmoid
inline Var sigmoid(const Var& a) {
    double val = 1.0 / (1.0 + std::exp(-a.val()));
    return Var(global_tape.push(val, {a.idx}, {val * (1.0 - val)}, "Sigmoid"));
}

// 16. Tanh
inline Var tanh(const Var& a) {
    double val = std::tanh(a.val());
    return Var(global_tape.push(val, {a.idx}, {1.0 - val * val}, "Tanh"));
}

// 17. Arcsin
inline Var asin(const Var& a) {
    return Var(global_tape.push(std::asin(a.val()), {a.idx}, {1.0 / std::sqrt(1.0 - a.val() * a.val())}, "Arcsin"));
}

// 18. Arccos
inline Var acos(const Var& a) {
    return Var(global_tape.push(std::acos(a.val()), {a.idx}, {-1.0 / std::sqrt(1.0 - a.val() * a.val())}, "Arccos"));
}