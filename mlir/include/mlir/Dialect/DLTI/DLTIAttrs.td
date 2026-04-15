//===- DLTIAttrs.td - DLTI dialect attributes definition --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DLTI_DLTIATTRS_TD
#define MLIR_DIALECT_DLTI_DLTIATTRS_TD

include "mlir/Dialect/DLTI/DLTI.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

class DLTIAttr<string name, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<DLTI_Dialect, name, traits, baseCppClass> { }

//===----------------------------------------------------------------------===//
// DataLayoutEntryAttr
//===----------------------------------------------------------------------===//

def DLTI_DataLayoutEntryAttr :
    DLTIAttr<"DataLayoutEntry", [DataLayoutEntryInterface]> {
  let summary = "An attribute to represent an entry of a data layout specification.";
  let description = [{
    A data layout entry attribute is a key-value pair where the key is a type or
    an identifier and the value is another attribute. These entries form a data
    layout specification.
  }];
  let parameters = (ins
    "DataLayoutEntryKey":$key, "Attribute":$value
  );
  // TODO: We do not generate storage class because llvm::PointerUnion
  // does not work with hash_key method.
  let genStorageClass = 0;
  let mnemonic = "dl_entry";
  let genVerifyDecl = 0;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Returns the entry with the given key and value.
    static DataLayoutEntryAttr get(StringAttr key, Attribute value);
    static DataLayoutEntryAttr get(MLIRContext *context, Type key, Attribute value);
    static DataLayoutEntryAttr get(Type key, Attribute value);
  }];
}

//===----------------------------------------------------------------------===//
// DataLayoutSpecAttr
//===----------------------------------------------------------------------===//

def DLTI_DataLayoutSpecAttr :
    DLTIAttr<"DataLayoutSpec", [DataLayoutSpecInterface]> {
  let summary = "An attribute to represent a data layout specification.";
  let description = [{
    A data layout specification is a list of entries that specify (partial) data
    layout information. It is expected to be attached to operations that serve
    as scopes for data layout requests.
  }];
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface", "">:$entries
  );
  let mnemonic = "dl_spec";
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Combines this specification with `specs`, enclosing specifications listed
    /// from outermost to innermost. This overwrites the older entries with the
    /// same key as the newer entries if the entries are compatible. Returns null
    /// if the specifications are not compatible.
    DataLayoutSpecAttr combineWith(ArrayRef<DataLayoutSpecInterface> specs) const;

    /// Returns the endiannes identifier.
    StringAttr getEndiannessIdentifier(MLIRContext *context) const;

    /// Returns the default memory space identifier.
    StringAttr getDefaultMemorySpaceIdentifier(MLIRContext *context) const;

    /// Returns the alloca memory space identifier.
    StringAttr getAllocaMemorySpaceIdentifier(MLIRContext *context) const;

    /// Returns the mangling mode identifier.
    StringAttr getManglingModeIdentifier(MLIRContext *context) const;

    /// Returns the program memory space identifier.
    StringAttr getProgramMemorySpaceIdentifier(MLIRContext *context) const;

    /// Returns the global memory space identifier.
    StringAttr getGlobalMemorySpaceIdentifier(MLIRContext *context) const;

    /// Returns the stack alignment identifier.
    StringAttr getStackAlignmentIdentifier(MLIRContext *context) const;

    /// Returns the attribute associated with the key.
    FailureOr<Attribute> query(DataLayoutEntryKey key) {
      return ::llvm::cast<mlir::DataLayoutSpecInterface>(*this).queryHelper(key);
    }
  }];
}

//===----------------------------------------------------------------------===//
// MapAttr
//===----------------------------------------------------------------------===//

def DLTI_MapAttr : DLTIAttr<"Map", [DLTIQueryInterface, QueryableAttrInterface]> {
  let summary = "A mapping of DLTI-information by way of key-value pairs";
  let description = [{
    A Data Layout and Target Information map is a list of entries effectively
    encoding a dictionary, mapping DLTI-related keys to DLTI-related values.

    This attribute's main purpose is to facilate querying IR for arbitrary
    key-value associations that encode DLTI. Facility functions exist to perform
    recursive lookups on nested DLTI-map/query interface-implementing
    attributes.

    Consider the following flat encoding of a single-key dictionary
    ```
    #dlti.map<"CPU::cache::L1::size_in_bytes" = 65536 : i32>>
    ```
    versus nested maps, which make it possible to obtain sub-dictionaries of
    related information (with the following example making use of other
    attributes that also implement the `DLTIQueryInterface`):
    ```
    #dlti.target_system_spec<"CPU" =
      #dlti.target_device_spec<"cache" =
        #dlti.map<"L1" = #dlti.map<"size_in_bytes" = 65536 : i32>,
                  "L1d" = #dlti.map<"size_in_bytes" = 32768 : i32> >>>
    ```

    With the flat encoding, the implied structure of the key is ignored, that is
    the only successful query (as expressed in the Transform Dialect) is:
    `transform.dlti.query ["CPU::cache::L1::size_in_bytes"] at %op`,
    where `%op` is a handle to an operation which associates the flat-encoding
    `#dlti.map` attribute.

    For querying nested dictionaries, the relevant keys need to be separately
    provided. That is, if `%op` is an handle to an op which has the nesting
    `#dlti.target_system_spec`-attribute from above attached, then
    `transform.dlti.query ["CPU","cache","L1","size_in_bytes"] at %op` gives
    back the first leaf value contained. To access the other leaf, we need to do
    `transform.dlti.query ["CPU","cache","L1d","size_in_bytes"] at %op`.
   }];
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface", "">:$entries
  );
  let mnemonic = "map";
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Returns the attribute associated with the key.
    FailureOr<Attribute> query(DataLayoutEntryKey key) {
      for (DataLayoutEntryInterface entry : getEntries())
        if (entry.getKey() == key)
            return entry.getValue();
      return ::mlir::failure();
    }
    FailureOr<Attribute> query(AttrQueryContext *context) {
      QueryableKey key = context->peekKey();
      // DataLayoutEntryKey dlKey;
      // if (auto keyAttr = key.dyn_cast<Attribute>()) {
      //   if (auto keyStrAttr = keyAttr.dyn_cast<StringAttr>()) {
      //     dlKey = DataLayoutEntryKey(keyStrAttr);
      //   } else {
      //     return ::mlir::failure();
      //   }
      // } else if (auto keyType = key.dyn_cast<Type>()) {
      //   dlKey = DataLayoutEntryKey(keyType);
      // }
      for (DataLayoutEntryInterface entry : getEntries()) {
        //if (entry.getKey() == dlKey) {
        if (entry.getKey() == key) {
          context->consumeKey(key);
          return entry.getValue();
        }
      }

      return ::mlir::failure();
    }
  }];
}

def DLTI_MatmulRegisterBlockingModelAttr : DLTIAttr<"MatmulRegisterBlockingModel", [QueryableAttrInterface]> {
  let summary = "TODO";
  let description = [{
    TODO
   }];
  let mnemonic = "matmul_register_blocking_model";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    FailureOr<Attribute> query(AttrQueryContext *context);
  }];
}

def DLTI_MatmulCostModelAttr : DLTIAttr<"MatmulCostModel", [QueryableAttrInterface]> {
  let summary = "TODO";
  let description = [{
    TODO
   }];
  let mnemonic = "matmul_cost_model";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    FailureOr<Attribute> query(AttrQueryContext *context);
  }];
}
  //let extraClassDeclaration = [{
  //  FailureOr<Attribute> query(AttrQueryContext *context) {
  //    QueryableKey modeKey = context->peekKey(/*index=*/0);
  //    int mode = 0;

  //    //if (auto modeStrAttr = dyn_cast<StringAttr>(dyn_cast<Attribute>(modeKey))) {
  //    if (auto modeStrAttr = dyn_cast<StringAttr>(modeKey)) {
  //      if (modeStrAttr.getValue() == "register block") {
  //        mode = 1;
  //      } else if (modeStrAttr.getValue() == "cache tile") {
  //        mode = 2;
  //      }
  //    }
  //
  //    //

  //    if (mode == 0) return ::mlir::failure();

  //    QueryableKey operatorKey = context->peekKey(/*index=*/1);
  //    //if (auto operatorStrAttr = dyn_cast<StringAttr>(dyn_cast<Attribute>(operatorKey))) {
  //    if (auto operatorStrAttr = dyn_cast<StringAttr>(operatorKey)) {
  //      if (operatorStrAttr.getValue() != "matmul") {
  //        return ::mlir::failure();
  //      }
  //    } else {
  //      return ::mlir::failure();
  //    }

  //    if (mode == 1) {
  //      QueryableKey dimKey = context->peekKey(/*index=*/2);
  //      StringRef dimStr;
  //      //if (auto dimStrAttr = dyn_cast<StringAttr>(dyn_cast<Attribute>(dimKey))) {
  //      if (auto dimStrAttr = dyn_cast<StringAttr>(dimKey)) {
  //        dimStr = dimStrAttr.getValue();
  //        if (dimStr != "M" && dimStr != "N") {
  //          return ::mlir::failure();
  //        }
  //      } else {
  //        return ::mlir::failure();
  //      }

  //      QueryableKey elemTypeKey = context->peekKey(/*index=*/3);
  //      Type elemType;
  //      if (!(elemType = dyn_cast<Type>(elemTypeKey))) {
  //        return ::mlir::failure();
  //      }

  //      QueryableKey k0 = StringAttr::get(getContext(), "CPU");
  //      QueryableKey k1 = StringAttr::get(getContext(), "vector unit");
  //      QueryableKey k2 = StringAttr::get(getContext(), "registers fit");
  //      FailureOr<Attribute> maybeSpaceInRegisters = \
  //          context->query({k0, k1, k2, elemType});

  //      if (failed(maybeSpaceInRegisters)) {
  //        return ::mlir::failure();
  //      }

  //      auto spaceInRegisters = dyn_cast<IntegerAttr>(*maybeSpaceInRegisters).getInt();

  //      Type i64 = IntegerType::get(getContext(), 64);
  //      if (spaceInRegisters >= 32 && spaceInRegisters < 64) {
  //        context->consumeKey();
  //        context->consumeKey();
  //        context->consumeKey();
  //        context->consumeKey();
  //        if (dimStr == "M") {
  //          return IntegerAttr::get(i64, 5);
  //        } else if (dimStr == "N") {
  //          return IntegerAttr::get(i64, 6);
  //        }
  //      } else if (spaceInRegisters >= 64 && spaceInRegisters < 128) {
  //        context->consumeKey();
  //        context->consumeKey();
  //        context->consumeKey();
  //        context->consumeKey();
  //        if (dimStr == "M") {
  //          return IntegerAttr::get(i64, 6);
  //        } else if (dimStr == "N") {
  //          return IntegerAttr::get(i64, 16);
  //        }
  //      }
  //    }

  //    if (mode == 2) {
  //      // TODO: caching tile cost model
  //      return ::mlir::failure();
  //    }

  //    return ::mlir::failure();
  //  }
  //}];

// def DLTI_MatmulCostModelAttr_OLD : DLTIAttr<"MatmulCostModel", [QueryableAttrInterface]> {
//   let summary = "TODO";
//   let description = [{
//     TODO
//    }];
//   let mnemonic = "matmul_cost_model";
//   let genVerifyDecl = 1;
//   let extraClassDeclaration = [{
//     FailureOr<Attribute> query(AttrQueryContext *context) {
//       QueryableKey level = context->peekKey(index=0);
//       QueryableKey action = context->peekKey(index=1);
//
//       int mode = 0;
//
//       if (auto levelStrAttr = level.dyn_cast<Attribute>().dyn_cast<StringAttr>()) {
//         if (auto actionStrAttr = action.dyn_cast<Attribute>().dyn_cast<StringAttr>()) {
//           if (levelStrAttr.getValue() == "cache" &&
//               actionStrAttr.getValue() == "tiling") {
//           mode = 1;
//         } else if (levelStrAttr.getValue() == "register" &&
//                    actionStrAttr.getValue() == "blocking") {
//           mode = 2;
//         }
//       }
//       if (mode == 0)
//         return ::mlir::failure();
//
//       if (mode == 1) {
//         QueryableKey typeAKey = context->peekKey(index=2);
//         QueryableKey typeBKey = context->peekKey(index=3);
//         ShapedType typeA, typeB;
//         if (!(typeA = typeAKey.dyn_cast<Type>().dyn_cast<ShapedType>()) ||
//             !(typeB = typeBKey.dyn_cast<Type>().dyn_cast<ShapedType>())) {
//           return ::mlir::failure();
//         }
//
//         ArrayRef<int64_t> shapeA = typeA.getShape();
//         ArrayRef<int64_t> shapeB = typeB.getShape();
//         int64_t m = shapeA[0];
//         int64_t n = shapeB[1];
//         int64_t k = shapeA[1];
//
//         FailureOr<Attribute> maybeCacheInfo = context.query({"CPU", "cache", "L1", "size_in_bytes"});
//         if (maybeCacheInfo.failed()) {
//           return ::mlir::failure();
//         }
//
//         auto l1CacheSize = maybeCacheInfo.get().dyn_cast<IntegerAttr>().getValue();
//
//         if (m * k + n * k + m * n <= l1CacheSize) {
//           context.consumeKey();
//           context.consumeKey();
//           context.consumeKey();
//           context.consumeKey();
//           return IntegerAttr::get(getContext(), 0);
//         }
//
//       }
//
//       if (mode == 2) {
//         FailureOr<Attribute> maybeRegisterInfo = context.query({"CPU", "vector_unit", "num_registers"});
//       }
//
//       QueryableKey level = context->peekKey(index=0);
//
//       return ::mlir::failure();
//     }
//   }];
// }

//===----------------------------------------------------------------------===//
// TargetSystemSpecAttr
//===----------------------------------------------------------------------===//

def DLTI_TargetSystemSpecAttr :
    DLTIAttr<"TargetSystemSpec", [TargetSystemSpecInterface,QueryableAttrInterface]> {
  let summary = "An attribute to represent target system specification.";
  let description = [{
    A system specification describes the overall system containing
    multiple devices, with each device having a unique ID (string)
    and its corresponding TargetDeviceSpec object.

    Example:
    ```
    dlti.target_system_spec =
     #dlti.target_system_spec<
      "CPU" = #dlti.target_device_spec<
        "L1_cache_size_in_bytes" = 4096: ui32>,
      "GPU" = #dlti.target_device_spec<
        "max_vector_op_width" = 64 : ui32>,
      "XPU" = #dlti.target_device_spec<
        "max_vector_op_width" = 4096 : ui32>>
    ```

    The verifier checks that keys are strings and pointed to values implement
    DLTI's TargetDeviceSpecInterface.
  }];
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface">:$entries
  );
  let mnemonic = "target_system_spec";
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Return the device specification that matches the given device ID
    std::optional<TargetDeviceSpecInterface>
    getDeviceSpecForDeviceID(
      TargetSystemSpecInterface::DeviceID deviceID);

    /// Returns the attribute associated with the key.
    FailureOr<Attribute> query(DataLayoutEntryKey key) const {
      return ::llvm::cast<mlir::TargetSystemSpecInterface>(*this).queryHelper(key);
    }
    FailureOr<Attribute> query(AttrQueryContext *context) {
      QueryableKey key = context->peekKey();
      for (DataLayoutEntryInterface entry : getEntries()) {
        //if (entry.getKey() == dlKey) {
        if (entry.getKey() == key) {
          context->consumeKey(key);
          return entry.getValue();
        }
      }

      return ::mlir::failure();
    }
  }];
  let extraClassDefinition = [{
    std::optional<TargetDeviceSpecInterface>
    $cppClass::getDeviceSpecForDeviceID(
        TargetSystemSpecInterface::DeviceID deviceID) {
      for (const auto& entry : getEntries()) {
        if (entry.getKey() == DataLayoutEntryKey(deviceID))
          if (auto deviceSpec =
              ::llvm::dyn_cast<TargetDeviceSpecInterface>(entry.getValue()))
            return deviceSpec;
      }
      return std::nullopt;
    }
  }];
}

//===----------------------------------------------------------------------===//
// TargetDeviceSpecAttr
//===----------------------------------------------------------------------===//

def DLTI_TargetDeviceSpecAttr :
    DLTIAttr<"TargetDeviceSpec", [TargetDeviceSpecInterface,QueryableAttrInterface]> {
  let summary = "An attribute to represent target device specification.";
  let description = [{
    Each device specification describes a single device and its
    hardware properties. Each device specification can contain any number
    of optional hardware properties (e.g., max_vector_op_width below).

    Example:
    ```
    #dlti.target_device_spec<"max_vector_op_width" = 64 : ui32>
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface">:$entries
  );
  let mnemonic = "target_device_spec";
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the attribute associated with the key.
    FailureOr<Attribute> query(DataLayoutEntryKey key) const {
      return ::llvm::cast<mlir::TargetDeviceSpecInterface>(*this).queryHelper(key);
    }
    FailureOr<Attribute> query(AttrQueryContext *context) {
      QueryableKey key = context->peekKey();
      for (DataLayoutEntryInterface entry : getEntries()) {
        //if (entry.getKey() == dlKey) {
        if (entry.getKey() == key) {
          context->consumeKey(key);
          return entry.getValue();
        }
      }

      return ::mlir::failure();
    }
  }];
}

#endif  // MLIR_DIALECT_DLTI_DLTIATTRS_TD
