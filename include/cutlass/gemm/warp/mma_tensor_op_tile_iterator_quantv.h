/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    ///// shape n influence the choose of mma tensor op tile iterator
    int ShapeN,
    /// Operand identity
    Operand Operand,
    /// Data type of A elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1>
class MmaTensorOpMultiplicandTileIterator_QuantV;

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator_QuantV<
    Shape_, 128, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                   64>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  // Layout = <16, 64>
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, 64>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner), 
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using RealFragment =
     Array<int8_t, Shape::kContiguous * InstructionShape::kStrided / kThreads>;
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

public:
  int lane_id_;
  int iteration_;

  int source_int_offset_;
  int source_char_offset_;
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV(
    TensorRef const &ref, 
    int lane_id
  ):
    stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0),
    k_group_idx_(0) {

    iteration_ = 0;
    lane_id_ = lane_id;

    auto warp_id = threadIdx.y;

    int lane_switch[2];
    // lane_switch[0] = lane_id % 4 + (warp_id / 2) * 16;
    lane_switch[0] = lane_id % 4 + (warp_id % 2) * 16;
    lane_switch[1] = lane_switch[0] + 8;

    source_int_offset_ = (lane_id / 4) / 4;
    source_char_offset_ = (lane_id / 4) % 4;

    source_int_offset_ += (warp_id / 2) * 32;

    // stride_: 8
    // Layout::kElementsPerAccess: 8
    // kPointerCount: 2
    // Policy::LdsmShape::kContiguous: 4
    // Layout::PartitionShape::kStrided: 4
    // Policy::kGroupsPerTile: 4
    // Policy::LdsmIterations::kContiguous: 1
    // Policy::LdsmIterations::kStrided: 1

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      lane_id = lane_switch[i];

      int quad_pair = (lane_id >> 3);
      int quad_quad = (lane_id >> 4);
      int lane_in_quad = (lane_id & 3);
      int lane_in_quad_pair = (lane_id & 7);
      int lane_in_quad_quad = (lane_id & 15);

      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

      if (Policy::LdsmShape::kContiguous == 4) {
        // Matrix multiply 1688 A/B
        // Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
        // Four blocks are next to each other in the contiguous dimension.
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      }
      else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816 A
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2));
        access_contiguous_idx =
            ((quad_pair & 1) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 B
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_quad ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kContiguous == 1) {
        // Matrix multiply 16832.SP B
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2); 
        access_contiguous_idx = lane_in_quad; 
        access_strided_idx = lane_id; 
      }

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      auto batch_id = blockIdx.z;
      auto head_id = blockIdx.y;
      auto warp_id = threadIdx.y;
      // printf("batch_id: %d, head_id: %d, warp_id: %d, lane_id_: %d, kPointerCountID: %d, access_contiguous: %d, access_strided: %d, partition_contiguous_idx: %d, access_contiguous_idx: %d\n", batch_id, head_id, warp_id, lane_id_, i, access_contiguous, access_strided, partition_contiguous_idx, access_contiguous_idx);

      pointer_[i] = reinterpret_cast<AccessType *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
      if (tile_offset.contiguous() % 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    // printf("tile_offset.strided() * InstructionShape::kStrided: %d, stride_: %d, Layout::kElementsPerAccess: %d, contiguous_offset: %d, Shape::kContiguous: %d\n", tile_offset.strided() * InstructionShape::kStrided, stride_, Layout::kElementsPerAccess, contiguous_offset, Shape::kContiguous);

    int offset = ((tile_offset.strided() * InstructionShape::kStrided) / 2) *
                     stride_ * Layout::kElementsPerAccess +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator++() {
    iteration_ += 1;
    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(RealFragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    // only for debug
    int8_t *fetch_ptr = reinterpret_cast<int8_t *>(&frag);
    
    AccessType *source_ptr = pointer_[0];
    char *source_byte_ptr = reinterpret_cast<char *>(source_ptr) + byte_offset + byte_offset_;
    int load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_);
    int8_t *load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);

    frag[0] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 16);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[1] = *(load_int_ptr + source_char_offset_);
    
    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 2);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[2] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 18);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[3] = *(load_int_ptr + source_char_offset_);

    source_ptr = pointer_[1];
    source_byte_ptr = reinterpret_cast<char *>(source_ptr) + byte_offset + byte_offset_;
    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[4] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 16);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[5] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 2);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[6] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 18);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[7] = *(load_int_ptr + source_char_offset_);

    auto batch_id = blockIdx.z;
    auto head_id = blockIdx.y;
    auto warp_id = threadIdx.y;
    // printf("batch_id: %d, head_id: %d, warp_id: %d, lane_id: %d, iteration_: %d, sx[0]: %f, sx[1]: %f, sx[2]: %f, sx[3]: %f, sx[4]: %f, sx[5]: %f, sx[6]: %f, sx[7]: %f\n", batch_id, head_id, warp_id, lane_id_, iteration_, float(*((int8_t *)fetch_ptr+0)), float(*((int8_t *)fetch_ptr+1)), float(*((int8_t *)fetch_ptr+2)), float(*((int8_t *)fetch_ptr+3)), float(*((int8_t *)fetch_ptr+4)), float(*((int8_t *)fetch_ptr+5)), float(*((int8_t *)fetch_ptr+6)), float(*((int8_t *)fetch_ptr+7)));
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = 
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess + 
      tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};
////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator_QuantV<
    Shape_, 64, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                   64>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  // Layout = <16, 64>
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, 64>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner), 
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using RealFragment =
     Array<int8_t, Shape::kContiguous * InstructionShape::kStrided / kThreads>;
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

public:
  int lane_id_;
  int iteration_;

  int source_int_offset_;
  int source_char_offset_;
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV(
    TensorRef const &ref, 
    int lane_id
  ):
    stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0),
    k_group_idx_(0) {

    iteration_ = 0;
    lane_id_ = lane_id;

    auto warp_id = threadIdx.y;

    int lane_switch[2];
    lane_switch[0] = lane_id % 4 + (warp_id / 2) * 16;
    lane_switch[1] = lane_switch[0] + 8;

    source_int_offset_ = (lane_id / 4) / 4;
    source_char_offset_ = (lane_id / 4) % 4;

    // stride_: 8
    // Layout::kElementsPerAccess: 8
    // kPointerCount: 2
    // Policy::LdsmShape::kContiguous: 4
    // Layout::PartitionShape::kStrided: 4
    // Policy::kGroupsPerTile: 4
    // Policy::LdsmIterations::kContiguous: 1
    // Policy::LdsmIterations::kStrided: 1

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      lane_id = lane_switch[i];

      int quad_pair = (lane_id >> 3);
      int quad_quad = (lane_id >> 4);
      int lane_in_quad = (lane_id & 3);
      int lane_in_quad_pair = (lane_id & 7);
      int lane_in_quad_quad = (lane_id & 15);

      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

      if (Policy::LdsmShape::kContiguous == 4) {
        // Matrix multiply 1688 A/B
        // Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
        // Four blocks are next to each other in the contiguous dimension.
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      }
      else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816 A
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2));
        access_contiguous_idx =
            ((quad_pair & 1) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 B
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_quad ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kContiguous == 1) {
        // Matrix multiply 16832.SP B
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2); 
        access_contiguous_idx = lane_in_quad; 
        access_strided_idx = lane_id; 
      }

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      auto batch_id = blockIdx.z;
      auto head_id = blockIdx.y;
      auto warp_id = threadIdx.y;
      // printf("batch_id: %d, head_id: %d, warp_id: %d, lane_id_: %d, kPointerCountID: %d, access_contiguous: %d, access_strided: %d, partition_contiguous_idx: %d, access_contiguous_idx: %d\n", batch_id, head_id, warp_id, lane_id_, i, access_contiguous, access_strided, partition_contiguous_idx, access_contiguous_idx);

      pointer_[i] = reinterpret_cast<AccessType *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
      if (tile_offset.contiguous() % 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    // printf("tile_offset.strided() * InstructionShape::kStrided: %d, stride_: %d, Layout::kElementsPerAccess: %d, contiguous_offset: %d, Shape::kContiguous: %d\n", tile_offset.strided() * InstructionShape::kStrided, stride_, Layout::kElementsPerAccess, contiguous_offset, Shape::kContiguous);

    int offset = ((tile_offset.strided() * InstructionShape::kStrided) / 2) *
                     stride_ * Layout::kElementsPerAccess +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator++() {
    iteration_ += 1;
    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(RealFragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    // only for debug
    int8_t *fetch_ptr = reinterpret_cast<int8_t *>(&frag);
    
    AccessType *source_ptr = pointer_[0];
    char *source_byte_ptr = reinterpret_cast<char *>(source_ptr) + byte_offset + byte_offset_;
    int load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_);
    int8_t *load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);

    frag[0] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 16);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[1] = *(load_int_ptr + source_char_offset_);
    
    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 2);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[2] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 18);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[3] = *(load_int_ptr + source_char_offset_);

    source_ptr = pointer_[1];
    source_byte_ptr = reinterpret_cast<char *>(source_ptr) + byte_offset + byte_offset_;
    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[4] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 16);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[5] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 2);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[6] = *(load_int_ptr + source_char_offset_);

    load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_ + 18);
    load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
    frag[7] = *(load_int_ptr + source_char_offset_);

    auto batch_id = blockIdx.z;
    auto head_id = blockIdx.y;
    auto warp_id = threadIdx.y;
    // printf("batch_id: %d, head_id: %d, warp_id: %d, lane_id: %d, iteration_: %d, sx[0]: %f, sx[1]: %f, sx[2]: %f, sx[3]: %f, sx[4]: %f, sx[5]: %f, sx[6]: %f, sx[7]: %f\n", batch_id, head_id, warp_id, lane_id_, iteration_, float(*((int8_t *)fetch_ptr+0)), float(*((int8_t *)fetch_ptr+1)), float(*((int8_t *)fetch_ptr+2)), float(*((int8_t *)fetch_ptr+3)), float(*((int8_t *)fetch_ptr+4)), float(*((int8_t *)fetch_ptr+5)), float(*((int8_t *)fetch_ptr+6)), float(*((int8_t *)fetch_ptr+7)));
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = 
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess + 
      tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    ///// shape n influence the choose of mma tensor op tile iterator
    int ShapeN,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator_QuantV<
    Shape_, ShapeN, Operand_, Element_,
    cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator_QuantV<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, ShapeN, kOperand, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            int(128 / sizeof(Element_))>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using RealFragment = typename Base::RealFragment;
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_QuantV & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(RealFragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    // TODO
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
