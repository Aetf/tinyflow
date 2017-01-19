/*!
 *  Copyright (c) 2016 by Aetf
 * \file parallelize.cpp
 * \brief Apply model/data parallel to marked nodes.
 */
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <algorithm>
#include <vector>

using std::vector;

namespace nnvm {
namespace pass {
namespace {

/*! \brief A use instance of an NodeEntry.
 *  Note that it uses the index from IndexedGraph, so will be invalidated if IndexedGraph changes.
 */
struct UseSite {
    uint32_t user_id;
    uint32_t entry_idx;
    bool is_graph_output;
    bool as_control_dep;

    UseSite(uint32_t user_id = 0, uint32_t entry_idx = 0,
            bool is_graph_output = false, bool as_control_dep = false)
        : user_id(user_id)
        , entry_idx(entry_idx)
        , is_graph_output(is_graph_output)
        , as_control_dep(as_control_dep)
    {}
};

// Build use sites info for all nodes by a DFS iteration
vector<vector<UseSite>> BuildUseSites(Graph &graph) {
    const IndexedGraph &idxg = graph.indexed_graph();
    vector<vector<UseSite>> useSiteInfo(idxg.num_nodes(), vector<UseSite>{});

    for (auto &e : graph.outputs) {
        auto source_id = idxg.node_id(e.node.get());
        useSiteInfo[source_id].emplace_back(0, e.index, true, false);
    }
    DFSVisit(graph.outputs, [&](const NodePtr &n) {
        auto nid = idxg.node_id(n.get());
        for (const auto &e : n->inputs) {
            auto source_id = idxg.node_id(e.node.get());
            useSiteInfo[source_id].emplace_back(nid, e.index, false, false);
        }
        for (const auto &dn : n->control_deps) {
            auto source_id = idxg.node_id(dn.get());
            useSiteInfo[source_id].emplace_back(nid, 0, false, true);
        }
    });

    return useSiteInfo;
}

// Substitute node with corresponding parallelized version
Graph Parallelize(Graph src) {
    Graph ret;
    vector<NodeEntry> outputs;

    auto useSites = BuildUseSites(src);

    DFSVisit(src.outputs, [](const NodePtr &n) {
        if (n->is_variable()
            || (n->op()->name != "matmul" && n->op()->name != "conv2d")) {
            // only support matmul and conv2d for now
            return;
        }
        auto it = n->attrs.dict.find("parallelism");
        if (it != n->attrs.dict.end()) {
            if (it->second == "model") {
                // model
            } else if (it->second == "data"){
                // data
            } else if (!it->second.empty()) {
                throw dmlc::Error("Unknown attribute value for parallelism");
            }
        }
    });
    return ret;
}

// register pass
NNVM_REGISTER_PASS(Parallelize)
.describe("Substitute suitable node with corresponding parallelized version")
.set_body(Parallelize)
.set_change_graph(true);

}  // namespace
}  // namespace pass
}  // namespace nnvm
