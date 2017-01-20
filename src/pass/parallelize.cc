/*!
 *  Copyright (c) 2016 by Aetf
 * \file parallelize.cpp
 * \brief Apply model/data parallel to marked nodes.
 */
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <algorithm>
#include <vector>

#include "../op_util.h"

using std::vector;

namespace tinyflow {
namespace pass {
namespace {

/*! \brief A use instance of an NodeEntry.
 *  Note that it uses the index from IndexedGraph, so will be invalidated if IndexedGraph changes.
 */
struct UseSite {
    uint32_t user_id;
    uint32_t user_input_idx;
    uint32_t src_output_idx;
    bool is_graph_output;
    bool as_control_dep;

    UseSite(uint32_t user_id = 0, uint32_t user_input_idx = 0, uint32_t src_output_idx = 0,
            bool is_graph_output = false, bool as_control_dep = false)
        : user_id(user_id)
        , user_input_idx(user_input_idx)
        , src_output_idx(src_output_idx)
        , is_graph_output(is_graph_output)
        , as_control_dep(as_control_dep)
    {}
};

// Build use sites info for all nodes by a DFS iteration
vector<vector<UseSite>> BuildUseSites(Graph &graph) {
    const IndexedGraph &idxg = graph.indexed_graph();
    vector<vector<UseSite>> useSiteInfo(idxg.num_nodes(), vector<UseSite>{});

    for (uint32_t i = 0; i != graph.outputs.size(); ++i) {
        const auto &e = graph.outputs[i];
        auto source_id = idxg.node_id(e.node.get());
        useSiteInfo[source_id].emplace_back(0, i, e.index, true, false);
    }
    DFSVisit(graph.outputs, [&](const NodePtr &n) {
        auto nid = idxg.node_id(n.get());
        for (uint32_t i = 0; i != n->inputs.size(); ++i) {
            const auto &e = n->inputs[i];
            auto source_id = idxg.node_id(e.node.get());
            useSiteInfo[source_id].emplace_back(nid, i, e.index, false, false);
        }
        for (uint32_t i = 0; i != n->control_deps.size(); ++i) {
            const auto &dn = n->control_deps[i];
            auto source_id = idxg.node_id(dn.get());
            useSiteInfo[source_id].emplace_back(nid, i, 0, false, true);
        }
    });

    return useSiteInfo;
}

void replaceUsage(const IndexedGraph &idxg, const UseSite &use, const NodeEntry &replacement) {
    // This is really ugly hack!
    auto userNode = const_cast<Node*>(idxg[use.user_id].source);
    userNode->inputs[use.user_input_idx] = replacement;
}

// Substitute node with corresponding parallelized version
Graph Parallelize(Graph src) {
    Graph ret;
    ret.outputs = src.outputs;

    auto useSites = BuildUseSites(src);
    auto &idxg = src.indexed_graph();

    DFSVisit(src.outputs, [&](const NodePtr &n) {
        if (n->is_variable()
            || (n->op()->name != "matmul" && n->op()->name != "conv2d")) {
            // only support matmul and conv2d for now
            return;
        }
        auto it = n->attrs.dict.find("parallelism");
        if (it != n->attrs.dict.end()) {
            if (it->second == "model") {
                // model
                if (n->op()->name == "matmul") {
                    // only support model for matmul
                    // get two inputs
                    auto netinput = n->inputs[0];
                    auto weight = n->inputs[1];
                    // split weight and netinput
                    auto nd_splited_weight = MakeNode("split", n->attrs.name + "/split0",
                                                      {weight}, {{"axis", "1"}, {"num_outputs", "2"}}).node;
                    auto nd_splited_netinput = MakeNode("split", n->attrs.name + "/split1",
                                                        {netinput},
                                                        {{"axis", "1"}, {"num_outputs", "2"}}).node;
                    // actual compute
                    auto new_attrs(n->attrs.dict);
                    new_attrs.erase("parallelism");
                    auto part0 = MakeNode("matmul", n->attrs.name + "/part0",
                                          {{nd_splited_netinput, 0, 0}, {nd_splited_weight, 0, 0}},
                                          new_attrs).node;
                    auto part1 = MakeNode("matmul", n->attrs.name + "/part1",
                                          {{nd_splited_netinput, 1, 0}, {nd_splited_weight, 1, 0}},
                                          new_attrs).node;
                    // preserve control deps
                    part0->control_deps = n->control_deps;
                    part1->control_deps = n->control_deps;
                    // merge output back, one merge operation per use site
                    const auto &sites = useSites[idxg.node_id(n.get())];
                    for (auto site : sites) {
                        if (!site.as_control_dep) {
                            auto merged_out = MakeNode("concat", n->attrs.name + "/merge0",
                            {{part0, 0, 0}, {part1, 0, 0}},
                            {{"axis", "1"}}).node;
                            if (!site.is_graph_output) {
                                replaceUsage(idxg, site, {merged_out, 0, 0});
                            } else {
                                ret.outputs[site.user_input_idx] = {merged_out, 0, 0};
                            }
                        }
                    }
                }
            } else if (it->second == "data"){
                // data
                if (n->op()->name == "conv2d") {
                    // only support data for conv2d
                    // get inputs
                    auto netinput = n->inputs[0];
                    auto new_inputs = n->inputs;
                    // split data
                    auto nd_splited_netinput = MakeNode("split", n->attrs.name + "/split0",
                                                        {netinput},
                                                        {{"axis", "0"}, {"num_outputs", "2"}}).node;
                    // actual compute
                    auto new_attrs(n->attrs.dict);
                    new_attrs.erase("parallelism");
                    new_inputs[0] = {nd_splited_netinput, 0, 0};
                    auto part0 = MakeNode("conv2d", n->attrs.name + "/part0",
                                          new_inputs,
                                          new_attrs).node;
                    new_inputs[0] = {nd_splited_netinput, 1, 0};
                    auto part1 = MakeNode("conv2d", n->attrs.name + "/part1",
                                          new_inputs,
                                          new_attrs).node;
                    // preserve control deps
                    part0->control_deps = n->control_deps;
                    part1->control_deps = n->control_deps;
                    // merge output back, one merge operation per use site
                    const auto &sites = useSites[idxg.node_id(n.get())];
                    for (auto site : sites) {
                        if (!site.as_control_dep) {
                            auto merged_out = MakeNode("concat", n->attrs.name + "/merge0",
                            {{part0, 0, 0}, {part1, 0, 0}},
                            {{"axis", "0"}}).node;
                            if (!site.is_graph_output) {
                                replaceUsage(idxg, site, {merged_out, 0, 0});
                            } else {
                                ret.outputs[site.user_input_idx] = {merged_out, 0, 0};
                            }
                        }
                    }
                }
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
