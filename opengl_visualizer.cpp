#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "autodiff.cpp"

using namespace std;

// --- Dynamic Graph State ---
int current_step = 0;
int max_steps = 0;
bool graph_initialized = false;

// Dynamic coordinate arrays
vector<float> node_x;
vector<float> node_y;

// --- Auto-Layout Algorithm ---
void compute_graph_layout() {
    size_t num_nodes = global_tape.nodes.size();
    node_x.resize(num_nodes);
    node_y.resize(num_nodes);
    
    vector<int> depth(num_nodes, 0);
    int max_depth = 0;
    
    // 1. Calculate topological depth for each node
    for (size_t i = 0; i < num_nodes; ++i) {
        int d = 0;
        for (size_t p : global_tape.nodes[i].parents) {
            if (depth[p] >= d) d = depth[p] + 1;
        }
        depth[i] = d;
        if (d > max_depth) max_depth = d;
    }

    // 2. Count nodes at each depth to space them vertically
    vector<int> depth_counts(max_depth + 1, 0);
    for (int d : depth) depth_counts[d]++;
    
    vector<int> current_depth_idx(max_depth + 1, 0);
    
    // 3. Assign X and Y coordinates
    for (size_t i = 0; i < num_nodes; ++i) {
        int d = depth[i];
        
        // Map X from -0.8 (left) to 0.8 (right)
        node_x[i] = -0.8f + (max_depth == 0 ? 0 : (1.6f * d / max_depth));
        
        // Map Y from -0.7 (bottom) to 0.7 (top)
        int count = depth_counts[d];
        int idx = current_depth_idx[d]++;
        if (count == 1) {
            node_y[i] = 0.0f; // Center if it's the only node at this depth
        } else {
            node_y[i] = -0.7f + (1.4f * idx / (count - 1));
        }
    }
}

// --- OpenGL Drawing Helpers ---
void drawText(float x, float y, void* font, const string& text, float r, float g, float b) {
    glColor3f(r, g, b);
    glRasterPos2f(x, y);
    for (char c : text) glutBitmapCharacter(font, c);
}

void drawCircle(float cx, float cy, float r, int num_segments, float red, float green, float blue) {
    glColor3f(red, green, blue);
    glBegin(GL_POLYGON);
    for (int i = 0; i < num_segments; i++) {
        float theta = 2.0f * 3.1415926f * float(i) / float(num_segments);
        glVertex2f(r * cosf(theta) + cx, r * sinf(theta) + cy);
    }
    glEnd();
    
    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < num_segments; i++) {
        float theta = 2.0f * 3.1415926f * float(i) / float(num_segments);
        glVertex2f(r * cosf(theta) + cx, r * sinf(theta) + cy);
    }
    glEnd();
}

void drawEdge(int from_idx, int to_idx, float r, float g, float b) {
    glColor3f(r, g, b);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glVertex2f(node_x[from_idx] + 0.1f, node_y[from_idx]); // Start at right edge of parent
    glVertex2f(node_x[to_idx] - 0.1f, node_y[to_idx]);     // End at left edge of child
    glEnd();
}

// --- Main Render Loop ---
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    size_t N = global_tape.nodes.size();
    
    drawText(-0.95f, 0.9f, GLUT_BITMAP_HELVETICA_18, "Dynamic Computational Graph", 1, 1, 1);
    drawText(-0.95f, 0.82f, GLUT_BITMAP_HELVETICA_12, "Press [SPACEBAR] to step through the autodiff passes.", 0.7f, 0.7f, 0.7f);
    
    string phase = "State: Initialized";
    if (current_step > 0 && current_step <= N) phase = "State: FORWARD PASS (Evaluating Values)";
    else if (current_step > N) phase = "State: BACKWARD PASS (Applying Chain Rule)";
    drawText(-0.95f, -0.9f, GLUT_BITMAP_HELVETICA_18, phase, 1, 1, 0);

    // 1. Draw Edges
    for (size_t i = 0; i < N; ++i) {
        for (size_t p : global_tape.nodes[i].parents) {
            drawEdge(p, i, 0.3f, 0.3f, 0.3f);
        }
    }

    // 2. Draw Nodes
    for (size_t i = 0; i < N; i++) {
        auto& node = global_tape.nodes[i];
        
        // Highlight logic
        float r = 0.2f, g = 0.2f, b = 0.2f; // Default state
        if (current_step > 0 && current_step <= N && (current_step - 1) == i) { r = 0.0f; g = 0.5f; b = 1.0f; } // Blue forward
        if (current_step > N && (max_steps - current_step) == i) { r = 1.0f; g = 0.0f; b = 0.0f; } // Red backward

        drawCircle(node_x[i], node_y[i], 0.1f, 32, r, g, b);
        drawText(node_x[i] - 0.05f, node_y[i] + 0.12f, GLUT_BITMAP_HELVETICA_12, node.op_name, 1, 1, 1);

        // Data Rendering
        string val_str = "Val: ?";
        if (current_step > i) {
            char buf[32]; snprintf(buf, sizeof(buf), "Val: %.2f", node.value); val_str = buf;
        }
        drawText(node_x[i] - 0.08f, node_y[i] - 0.02f, GLUT_BITMAP_HELVETICA_10, val_str, 0.5f, 1.0f, 0.5f);

        string grad_str = "Grad: 0.0";
        if (current_step > N && (max_steps - current_step) <= i) {
            char buf[32]; snprintf(buf, sizeof(buf), "Grad: %.2f", node.adjoint); grad_str = buf;
        }
        drawText(node_x[i] - 0.08f, node_y[i] - 0.06f, GLUT_BITMAP_HELVETICA_10, grad_str, 1.0f, 0.5f, 0.5f);
    }

    glutSwapBuffers();
}

// --- Interaction ---
void keyboard(unsigned char key, int x, int y) {
    if (key == ' ' || key == 13) {
        if (current_step < max_steps) {
            current_step++;
            size_t N = global_tape.nodes.size();
            
            // Trigger actual autodiff backward math in sync with the visualizer
            if (current_step == N + 1) {
                for (auto& n : global_tape.nodes) n.adjoint = 0.0;
                global_tape.nodes.back().adjoint = 1.0; 
            } else if (current_step > N + 1) {
                int node_idx = max_steps - current_step; 
                auto& curr = global_tape.nodes[node_idx + 1];
                for (size_t j = 0; j < curr.parents.size(); ++j) {
                    global_tape.nodes[curr.parents[j]].adjoint += curr.adjoint * curr.local_grads[j];
                }
            }
            glutPostRedisplay();
        }
    }
    if (key == 27) exit(0); // ESC to quit
}

int main(int argc, char** argv) {
    global_tape.clear();

    // Build a slightly more complex graph to test the auto-layout
    // Equation: z = (x * y) + sin(x) * exp(y)
    Var x(2.0); 
    Var y(1.5); 
    Var z = (x * y) + (sin(x) * exp(y)); 

    graph_initialized = true;
    compute_graph_layout(); // Run our new layout algorithm
    max_steps = global_tape.nodes.size() * 2; // Forward + Backward steps

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(1000, 700);
    glutCreateWindow("C++ OpenGL Autodiff Visualizer");
    glClearColor(0.08f, 0.08f, 0.08f, 1.0f);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    
    glutMainLoop();
    return 0;
}