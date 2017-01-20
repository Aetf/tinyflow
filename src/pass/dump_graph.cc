/*!
 *  Copyright (c) 2016 by Aetf
 * \file dump_graph.cpp
 * \brief Dump graph to stdout.
 */
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>

using std::cerr;
using std::cout;
using std::endl;
using std::ostringstream;

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

std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
    if(from.empty())
        return str;
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
    return str;
}

std::string id(const std::string &str) {
    return replaceAll(str, "/", "_");
}

Graph DotGraph(Graph src) {
    cout << "Begin DotGraph" << endl;

    auto *pout = &cerr;
    std::ofstream of;
    auto it = src.attrs.find("dotgraph_output");
    if (it != src.attrs.end()) {
        auto path = dmlc::get<std::string>(*it->second);
        of.open(path);
        if (of) {
            pout = &of;
        } else {
            LOG(INFO) << "dotgraph_output provided but invalid: " << path;
        }
    }
    auto &out = *pout;
    out << "digraph graphname {" << endl;

    int index = 0;
    DFSVisit(src.outputs, [&index, &src, &out](const NodePtr &n) {
        out << id(n->attrs.name);
        out << " [";
        if (!n->is_variable())
            out << "shape=box,";
        bool is_output = false;
        for (const auto &e : src.outputs) {
            if (e.node->attrs.name == n->attrs.name) {
                is_output = true;
                break;
            }
        }
        if (is_output) {
            out << "style=filled,fillcolor=gray,";
        }
        out << "label=\"";

        ostringstream oss;
        oss << "name: " << n->attrs.name << "\\l";
        oss << "op: ";
        if (n->is_variable())
            oss << "variable";
        else
            oss << n->op()->name;
        oss << "\\l";
        if (n->attrs.dict.size() != 0)
            oss << "attributes:" << "\\l";
        for (const auto &attr : n->attrs.dict) {
            oss << "    " << attr.first << ": " << attr.second << "\\l";
        }
        out << oss.str() << "\"" << "];" << endl;

        for (const auto &entry : n->inputs) {
            out << id(entry.node->attrs.name) << " -> " << id(n->attrs.name) << ";" << endl;
        }
        for (const NodePtr &d : n->control_deps) {
            out << id(d->attrs.name) << " -> " << id(n->attrs.name);
            out << " [style=dotted];" << endl;
        }
    });
    out << "}" << endl;
    cout << "End DotGraph" << endl;
    return src;
}

// register pass
NNVM_REGISTER_PASS(DumpGraph)
.describe("Return the same Graph, prints out the graph")
.set_body(DumpGraph)
.set_change_graph(false);

NNVM_REGISTER_PASS(DotGraph)
.describe("Return the same Graph, prints out the graph in DOT format")
.set_body(DotGraph)
.set_change_graph(false);

}  // namespace
}  // namespace pass
}  // namespace nnvm
