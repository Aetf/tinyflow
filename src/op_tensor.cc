// Copyright (c) 2016 by Contributors
// implementation of common tensor operators
#include <tinyflow/base.h>
#include <dmlc/parameter.h>
#include <nnvm/op_attr_types.h>
#include <cmath>
#include <utility>
#include "./op_util.h"

namespace tinyflow {

// shape given the ZeroParam
using namespace nnvm;

// shape parameter for zeros, ones
struct ZeroParam : public dmlc::Parameter<ZeroParam> {
  TShape shape;
  int dtype;
  DMLC_DECLARE_PARAMETER(ZeroParam) {
    DMLC_DECLARE_FIELD(shape).set_default(TShape());
    DMLC_DECLARE_FIELD(dtype).set_default(kFloat32);
  }
};
DMLC_REGISTER_PARAMETER(ZeroParam);

inline bool ZeroShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  const TShape& ts = dmlc::get<ZeroParam>(attrs.parsed).shape;
  if (ts.ndim() != 0) {
    SHAPE_ASSIGN(oshape->at(0), ts);
    return true;
  } else {
    return false;
  }
}

inline bool ZeroType(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int dtype = dmlc::get<ZeroParam>(attrs.parsed).dtype;
  DTYPE_ASSIGN(oattr->at(0), dtype);
  return true;
}


NNVM_REGISTER_OP_GROUP(ElementwiseOpAttr)
.set_attr<bool>("IsElementWise", true)
.set_attr<FInferShape>("FInferShape", SameShape);


NNVM_REGISTER_OP(zeros)
.describe("zeros")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ZeroParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FInferType>("FInferType", ZeroType);

NNVM_REGISTER_OP(zeros_like)
.describe("zeros_like")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape);

NNVM_REGISTER_OP(ones)
.describe("ones")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ZeroParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FInferType>("FInferType", ZeroType);


NNVM_REGISTER_OP(ones_like)
.describe("ones_like")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape);


NNVM_REGISTER_OP(normal)
.describe("normal distribution")
.set_num_inputs(0)
.set_attr_parser(ParamParser<ZeroParam>)
.set_attr<FInferShape>("FInferShape", ZeroShape)
.set_attr<FInferType>("FInferType", ZeroType);


NNVM_REGISTER_OP(equal)
.describe("Equal comparitor")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape);


NNVM_REGISTER_OP(__ewise_sum__)
.describe("ewise sum")
.set_num_inputs(nnvm::kVarg)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>(n->num_inputs(), ograds[0]);
});


NNVM_REGISTER_OP(__add_symbol__)
.describe("add two data together")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{ograds[0], ograds[0]};
});


NNVM_REGISTER_OP(__add_scalar__)
.describe("add symbol with scalar")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{ograds[0]};
});


NNVM_REGISTER_OP(__sub_symbol__)
.describe("do subtract")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("__mul_scalar__", n->attrs.name + "_grad_0",
                 {ograds[0]}, {{"scalar", "1"}}),
        MakeNode("__mul_scalar__", n->attrs.name + "_grad_1",
                 {ograds[0]}, {{"scalar", "-1"}}),
      };
});


NNVM_REGISTER_OP(__sub_scalar__)
.describe("subtract symbol with scalar")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{ograds[0]};
});


NNVM_REGISTER_OP(__rsub_scalar__)
.describe("subtract scalar with symbol")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("__mul_scalar__", n->attrs.name + "_grad_1",
                 {ograds[0]}, {{"scalar", "-1"}}),
        };
});


NNVM_REGISTER_OP(mul)
.add_alias("__mul_symbol__")
.describe("add two data together")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("mul", n->attrs.name + "_grad_0",
                 {ograds[0], n->inputs[1]}),
        MakeNode("mul", n->attrs.name + "_grad_1",
                 {ograds[0], n->inputs[0]})
      };
});


NNVM_REGISTER_OP(__mul_scalar__)
.describe("Multiply symbol with scalar")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("__mul_scalar__", n->attrs.name + "_grad_0",
                 {ograds[0]}, {{"scalar", n->attrs.dict["scalar"]}}),
      };
});


NNVM_REGISTER_OP(__div_symbol__)
.add_alias("div")
.describe("do division")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      NodeEntry n1 = MakeNode("mul", n->attrs.name + "_grad_sub_0",
                              {ograds[0], n->inputs[0]});
      NodeEntry n2 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_1",
                              {n1}, {{"scalar", "-1"}});
      NodeEntry n3 = MakeNode("mul", n->attrs.name + "_grad_sub_2",
                              {n->inputs[1], n->inputs[1]});
      return std::vector<NodeEntry>{
        MakeNode("__div_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], n->inputs[1]}),
        MakeNode("__div_symbol__", n->attrs.name + "_grad_1",
                 {n1, n2})
      };
});


NNVM_REGISTER_OP(__div_scalar__)
.describe("division symbol with scalar")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("__div_scalar__", n->attrs.name + "_grad_0",
                 {ograds[0]}, {{"scalar", n->attrs.dict["scalar"]}}),
      };
});


NNVM_REGISTER_OP(exp)
.describe("take elemtnwise exponation")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
        MakeNode("__mul_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], NodeEntry{n, 0, 0}})
      };
});


NNVM_REGISTER_OP(log)
.describe("take elemtnwise logarithm")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
        MakeNode("__div_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], n->inputs[0]})
      };
});


NNVM_REGISTER_OP(sqrt)
.describe("return square root of input")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    // 1 / (2 * sqrt(x)) == 1 / (2 * y)
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      NodeEntry n1 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_1",
                              {NodeEntry{n, 0, 0}}, {{"scalar", "2"}});
      return std::vector<NodeEntry>{
        MakeNode("__div_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], n1})
      };
});


NNVM_REGISTER_OP(__pow_symbol__)
.add_alias("pow")
.describe("take elmtnwise power between two tensor")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      // lhs: b*pow(a, b-1), rhs: pow(a, b)*ln(a)
      NodeEntry n0 = MakeNode("__add_scalar__", n->attrs.name + "_grad_sub_0",
                              {n->inputs[1]}, {{"scalar", "-1"}});
      NodeEntry n1 = MakeNode("pow", n->attrs.name + "_grad_sub_1",
                              {n->inputs[0], n0});
      NodeEntry d_lhs = MakeNode("mul", n->attrs.name + "_grad_sub_2",
                                 {n1, n->inputs[1]});
      NodeEntry n2 = MakeNode("log", n->attrs.name + "_grad_sub_3",
                              {n->inputs[0]});
      NodeEntry d_rhs = MakeNode("mul", n->attrs.name + "_grad_sub_4",
                                 {NodeEntry{n, 0, 0}, n2});
      return std::vector<NodeEntry>{
        MakeNode("__mul_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], d_lhs}),
        MakeNode("__mul_symbol__", n->attrs.name + "_grad_1",
                 {ograds[0], d_rhs})
      };

});


NNVM_REGISTER_OP(__rpow_scalar__)
.describe("take elmtnwise power between a number and a tensor")
.set_num_inputs(1)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      // pow(m, x) * ln(m)
      double num = std::stod(n->attrs.dict["scalar"]);
      NodeEntry n0 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_4",
                              {NodeEntry{n, 0, 0}}, {{"scalar", std::to_string(std::log(num))}});
      return std::vector<NodeEntry>{
        MakeNode("__mul_symbol__", n->attrs.name + "_grad_0",
                 {ograds[0], n0})
      };
});


NNVM_REGISTER_OP(matmul)
.describe("Matrix multiplication")
.set_num_inputs(2)
.set_attr<FInferShape>(
    "FInferShape", [](const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
      if (ishape->at(0).ndim() == 0) return false;
      if (ishape->at(1).ndim() == 0) return false;
      CHECK_EQ(ishape->at(0).ndim(), 2);
      CHECK_EQ(ishape->at(1).ndim(), 2);
      CHECK_EQ(ishape->at(0)[1], ishape->at(1)[0]);
      TShape target{ishape->at(0)[0], ishape->at(1)[1]};
      SHAPE_ASSIGN(oshape->at(0), target);
      return true;
    })
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_matmul_backward", n,
                               {ograds[0], n->inputs[0], n->inputs[1]});
    });


// simply register a bulk op for backward
NNVM_REGISTER_OP(_matmul_backward)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true);

struct ReduceParam : public dmlc::Parameter<ReduceParam> {
  Tuple<int> reduction_indices;
  DMLC_DECLARE_PARAMETER(ReduceParam) {
    DMLC_DECLARE_FIELD(reduction_indices).set_default(Tuple<int>());
  }
};
DMLC_REGISTER_PARAMETER(ReduceParam);


inline bool ReduceShape(const NodeAttrs& attrs,
                        std::vector<TShape> *ishape,
                        std::vector<TShape> *oshape) {
  const auto& axis
      = dmlc::get<ReduceParam>(attrs.parsed).reduction_indices;
  if (ishape->at(0).ndim() == 0) return false;
  if (axis.ndim() == 0) {
    SHAPE_ASSIGN(oshape->at(0), TShape{1});
  } else {
    TShape tmp = ishape->at(0);
    for (uint32_t idx : axis) {
      tmp[idx] = 0;
    }
    std::vector<uint32_t> ret;
    for (uint32_t x : tmp) {
      if (x != 0) ret.push_back(x);
    }
    if (ret.size() == 0) ret.push_back(1);
    SHAPE_ASSIGN(oshape->at(0), TShape(ret.begin(), ret.end()));
  }
  return true;
}


NNVM_REGISTER_OP(reduce_sum)
.describe("reduce sum")
.set_attr_parser(ParamParser<ReduceParam>)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", ReduceShape)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_reduce_sum_backward", n,
                               {ograds[0]}, n->attrs.dict);
    });


NNVM_REGISTER_OP(reduce_mean)
.describe("reduce mean")
.set_attr_parser(ParamParser<ReduceParam>)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", ReduceShape)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_reduce_mean_backward", n,
                               {ograds[0]}, n->attrs.dict);
    });


NNVM_REGISTER_OP_GROUP(ReduceBackwardIndeAttr)
.set_attr<nnvm::TIsBackward>("TIsBackward", true);


NNVM_REGISTER_OP(_reduce_sum_backward)
.set_num_inputs(1)
.set_num_outputs(1)
.include("ReduceBackwardIndeAttr");


NNVM_REGISTER_OP(_reduce_mean_backward)
.set_num_inputs(1)
.set_num_outputs(1)
.include("ReduceBackwardIndeAttr");


NNVM_REGISTER_OP(_argmax)
.set_attr_parser(ParamParser<ReduceParam>)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", ReduceShape);


struct ConcatParam : public dmlc::Parameter<ConcatParam> {
  uint32_t axis;
  DMLC_DECLARE_PARAMETER(ConcatParam) {
    DMLC_DECLARE_FIELD(axis).set_default(0);
  }
};
DMLC_REGISTER_PARAMETER(ConcatParam);

NNVM_REGISTER_OP(concat)
.describe("Concat tensors into one along one dimension.")
.set_num_outputs(1)
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<FInferShape>("FInferShape", [](const NodeAttrs &attrs,
                                         std::vector<TShape> *in_attrs,
                                         std::vector<TShape> *out_attrs) {
    auto axis = dmlc::get<ConcatParam>(attrs.parsed).axis;

    CHECK_GT(in_attrs->size(), 0);
    CHECK_EQ(out_attrs->size(), 1);

    auto num_inputs = in_attrs->size();

    auto out_shape = (*in_attrs)[0];
    auto ndim = out_shape.ndim();
    CHECK_LE(axis, ndim);

    for (uint32_t i = 1; i < num_inputs; ++i) {
        const auto &ishape = (*in_attrs)[i];
        if (ishape.ndim() == 0) {
            return false;
        }
        CHECK_EQ(ishape.ndim(), ndim);
        for (uint32_t j = 0; j != ndim; ++j) {
            if (j != axis && ishape[j] != out_shape[j]) {
                LOG(FATAL) << "Incompatible shape in " << attrs.name;
                return false;
            }
        }
        out_shape[axis] += ishape[axis];
    }
    if ((*out_attrs)[0].ndim() != 0) {
        CHECK_EQ(out_shape, (*out_attrs)[0]);
    } else {
        (*out_attrs)[0] = out_shape;
    }
    return true;
})
.set_attr<FGradient>("FGradient",
                     [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
    auto axis = dmlc::get<ConcatParam>(n->attrs.parsed).axis;
    CHECK_EQ(ograds.size(), 1);

    // split(ograds)
    auto bpnode = MakeNode("split", n->attrs.name + "_grad", {ograds[0]},
                           {
                               {"axis", std::to_string(axis)},
                               {"num_outputs", std::to_string(n->inputs.size())}
                           }).node;
    bpnode->control_deps.push_back(n);
    std::vector<NodeEntry> res;
    res.reserve(n->inputs.size());
    for (uint32_t i = 0; i != n->inputs.size(); ++i) {
        res.emplace_back(NodeEntry{bpnode, i, 0});
    }
    return res;
});

struct SplitParam : public dmlc::Parameter<SplitParam> {
  uint32_t axis;
  uint32_t num_outputs;
  DMLC_DECLARE_PARAMETER(SplitParam) {
    DMLC_DECLARE_FIELD(axis).set_default(0);
    DMLC_DECLARE_FIELD(num_outputs).set_default(2);
  }
};
DMLC_REGISTER_PARAMETER(SplitParam);

NNVM_REGISTER_OP(split)
.describe("Splits a tensor into num_split tensors along one dimension.")
.set_num_inputs(1)
.set_attr_parser(ParamParser<SplitParam>)
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    return dmlc::get<SplitParam>(attrs.parsed).num_outputs;
})
.set_attr<FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
    auto num_outputs = dmlc::get<SplitParam>(attrs.parsed).num_outputs;
    std::vector<std::pair<int, int>> res;
    res.reserve(num_outputs);
    for (uint32_t i = 0; i != num_outputs; ++i) {
        res.emplace_back(0, i);
    }
    return res;
})
.set_attr<FInferShape>("FInferShape", [](const NodeAttrs& attrs,
                                         std::vector<TShape> *in_attrs,
                                         std::vector<TShape> *out_attrs) {
    auto num_outputs = dmlc::get<SplitParam>(attrs.parsed).num_outputs;
    auto axis = dmlc::get<SplitParam>(attrs.parsed).axis;

    CHECK_EQ(in_attrs->size(), 1);
    CHECK_EQ(out_attrs->size(), num_outputs);

    if ((*in_attrs)[0].ndim() != 0) {
        TShape out_shape = (*in_attrs)[0];
        CHECK_LT(axis, out_shape.ndim());
        CHECK_EQ(out_shape[axis] % num_outputs, 0);
        out_shape[axis] /= num_outputs;
        for (auto &oshape : *out_attrs) {
            if (oshape.ndim() == 0) {
                oshape = out_shape;
            } else {
                if (oshape != out_shape) {
                    LOG(FATAL) << "Incompatible shape in node " << attrs.name
                               << " expected " << out_shape << " got " << oshape;
                    return false;
                }
            }
        }
        return true;
    }

    TShape out_shape;
    for (auto &oshape : *out_attrs) {
        if (oshape.ndim() != 0) {
            if (out_shape.ndim() != 0 && out_shape != oshape) {
                LOG(FATAL) << "";
                return false;
            }
            out_shape = oshape;
        } else if (out_shape.ndim() != 0) {
            oshape = out_shape;
        }
    }
    if (out_shape.ndim() != 0) {
        CHECK_LT(axis, out_shape.ndim());
        out_shape[axis] *= num_outputs;
          (*in_attrs)[0] = out_shape;
        return true;
    }
    return false;
})
.set_attr<FGradient>("FGradient",
                     [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
    auto axis = dmlc::get<SplitParam>(n->attrs.parsed).axis;
    // concat(ograds)
    auto bpnode = MakeNode("concat", n->attrs.name + "_grad",
                           ograds, {{"axis", std::to_string(axis)}}).node;
    bpnode->control_deps.push_back(n);
    return std::vector<NodeEntry>{ {bpnode, 0, 0} };
});

struct ReshapeParam : public dmlc::Parameter<ReshapeParam> {
  TShape shape;
  DMLC_DECLARE_PARAMETER(ReshapeParam) {
    DMLC_DECLARE_FIELD(shape).set_default(TShape{});
  }
};
DMLC_REGISTER_PARAMETER(ReshapeParam);

NNVM_REGISTER_OP(reshape)
.describe("reshape source to target shape")
.set_num_inputs(1)
.set_attr_parser(ParamParser<ReshapeParam>)
.set_attr<FInferShape>(
    "FInferShape", [] (const NodeAttrs& attrs,
                       std::vector<TShape> *ishape,
                       std::vector<TShape> *oshape) {
      // get parsed attribute
      const TShape& target = dmlc::get<ReshapeParam>(attrs.parsed).shape;
      (*oshape)[0] = target;
      if ((*ishape)[0].ndim() == 0) return false;
      CHECK_EQ((*ishape)[0].Size(), target.Size())
          << "Reshape op: source target shape mismatch";
      return true;
    })
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>("FGradient", [](const NodePtr &n, const std::vector<NodeEntry> &ograds) {
    LOG(INFO) << "Here";
    auto bpnode = MakeNode("reshape", n->attrs.name + "_grad", ograds, {{"shape", "[1]"}}).node;
    bpnode->control_deps.push_back(n);
    return std::vector<NodeEntry>{ {bpnode, 0, 0} };
});

}  // namespace tinyflow
