/*!
 *  Copyright (c) 2016 by Aetf
 * \file dump_graph.cpp
 * \brief Dump graph to stdout.
 */
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <algorithm>
#include <iostream>

using std::cout;
using std::endl;

namespace nnvm {
namespace pass {
namespace {

std::ostream &operator<<(std::ostream &out, const NodeEntry &entry) {
    return out << entry.node->attrs.name << ":" << entry.index << " version: " << entry.version;
}

// dump the graph to stdout
Graph DumpGraph(Graph src) {
    cout << "Begin DumpGraph" << endl;
    cout << "graph output num:" << src.outputs.size() << endl;
    for (const auto &entry : src.outputs) {
        cout << "    " << entry << endl;
    }
    cout << "All nodes" << endl;

    int index = 0;
    DFSVisit(src.outputs, [&index](const NodePtr &n) {
        cout << "node[" << index++ << "]=";
        if (n->is_variable())
            cout << "variable";
        else
            cout << n->op()->name;
        cout << "(name=" << n->attrs.name;
        cout << ", inputs=";

        cout << n->num_inputs() << ", outputs=";
        cout << n->num_outputs() << ")" << endl;
        cout << "  Inputs:";
        if (n->num_inputs() != n->inputs.size()) {
            cout << " size of the inputs mismatch!" << endl;
        } else {
            cout << endl;
        }
        for (const auto &entry : n->inputs) {
            cout << "    " << entry << endl;
        }
        cout << "  Control deps:" << endl;
        for (const NodePtr &d : n->control_deps) {
            cout << "    " << d->attrs.name << endl;
        }
        cout << "  Attributes:" << endl;
        for (const auto &attr : n->attrs.dict) {
            cout << "    " << attr.first << ": " << attr.second << endl;
        }
    });
    cout << "End DumpGraph" << endl;
    return src;
}

// register pass
NNVM_REGISTER_PASS(DumpGraph)
.describe("Return the same Graph, prints out the graph")
.set_body(DumpGraph)
.set_change_graph(false);

}  // namespace
}  // namespace pass
}  // namespace nnvm
