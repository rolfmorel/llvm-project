// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_REWRITE_H
#define MLIR_BINDINGS_PYTHON_REWRITE_H

// #include "mlir-c/Dialect/Transform.h"
#include "mlir-c/Rewrite.h"
// #include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
// #include "mlir/Bindings/Python/NanobindUtils.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

/// CRTP Base class for rewriter wrappers.
template <typename DerivedTy>
class PyRewriterBase {
public:
  PyRewriterBase(MlirRewriterBase rewriter)
      : base(rewriter),
        ctx(PyMlirContext::forContext(mlirRewriterBaseGetContext(base))) {}

  PyInsertionPoint getInsertionPoint() const {
    MlirBlock block = mlirRewriterBaseGetInsertionBlock(base);
    MlirOperation op = mlirRewriterBaseGetOperationAfterInsertion(base);

    if (mlirOperationIsNull(op)) {
      MlirOperation owner = mlirBlockGetParentOperation(block);
      auto parent = PyOperation::forOperation(ctx, owner);
      return PyInsertionPoint(PyBlock(parent, block));
    }

    return PyInsertionPoint(PyOperation::forOperation(ctx, op));
  }

  void replaceOp(MlirOperation op, MlirOperation newOp) {
    mlirRewriterBaseReplaceOpWithOperation(base, op, newOp);
  }

  void replaceOp(MlirOperation op, const std::vector<MlirValue> &values) {
    mlirRewriterBaseReplaceOpWithValues(base, op, values.size(), values.data());
  }

  void eraseOp(MlirOperation op) { mlirRewriterBaseEraseOp(base, op); }

  static void bind(nanobind::module_ &m) {
    nb::class_<DerivedTy>(m, DerivedTy::pyClassName)
        .def_prop_ro("ip", &PyRewriterBase::getInsertionPoint,
                     "The current insertion point of the PatternRewriter.")
        .def(
            "replace_op",
            [](DerivedTy &self, MlirOperation op, MlirOperation newOp) {
              self.replaceOp(op, newOp);
            },
            "Replace an operation with a new operation.", nb::arg("op"),
            nb::arg("new_op"),
            // clang-format off
              nb::sig("def replace_op(self, op: " MAKE_MLIR_PYTHON_QUALNAME("ir.Operation") ", new_op: " MAKE_MLIR_PYTHON_QUALNAME("ir.Operation") ") -> None")
            // clang-format on
            )
        .def(
            "replace_op",
            [](DerivedTy &self, MlirOperation op,
               const std::vector<MlirValue> &values) {
              self.replaceOp(op, values);
            },
            "Replace an operation with a list of values.", nb::arg("op"),
            nb::arg("values"),
            // clang-format off
              nb::sig("def replace_op(self, op: " MAKE_MLIR_PYTHON_QUALNAME("ir.Operation") ", values: list[" MAKE_MLIR_PYTHON_QUALNAME("ir.Value") "]) -> None")
            // clang-format on
            )
        .def("erase_op", &PyRewriterBase::eraseOp, "Erase an operation.",
             nb::arg("op"),
             // clang-format off
                nb::sig("def erase_op(self, op: " MAKE_MLIR_PYTHON_QUALNAME("ir.Operation") ") -> None")
             // clang-format on
        );
  }

private:
  MlirRewriterBase base;
  PyMlirContextRef ctx;
};

void populateRewriteSubmodule(nanobind::module_ &m);
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_REWRITE_H
