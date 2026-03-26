#include <iostream>
#include "autodiff.cpp"

// creates .dot to use for Graphviz
int main() {
    // Clear the tape to ensure a fresh graph
    global_tape.clear();

    // 1. Define inputs
    Var x(2.0);
    Var y(3.0);

    // 2. Build the computational graph: z = (x * y) + sin(x)
    Var z = (x * y) + sin(x);

    // 3. Run reverse-mode autodiff
    z.backward();

    // 4. Export the graph to a file
    global_tape.export_graphviz("computational_graph.dot");

    return 0;
}