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
class MmaTensorOpMultiplicandTileIterator_Scale;

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
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
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator_Scale<
    Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  int lane_id_[4];
  int origin_lane_id_;
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
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
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations =
        layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
                                        LdsmShape::kStrided>;

    ///
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using RealFragment = Array<int8_t, Shape::kStrided * InstructionShape::kContiguous / kThreads>;
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_[4];

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

  int source_int_offset_;
  int source_char_offset_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale()
      : pointer_(nullptr),
        sections_(0),
        stride_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        k_group_idx_(0) {
    // Warp level iterator at most use double buffer to hide latency.  If there
    // are more than 2 sections, every stage should have more than 1 section.

    // Turing silicon requires all 32 threads in a warp provide valid addresses
    // even for LDSM.1 and LDSM.2
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750))
    lane_id = lane_id % (Policy::LdsmShape::kCount * Policy::kLdsmOpInner);
#endif
    origin_lane_id_ = lane_id;

    source_int_offset_ = (origin_lane_id_ % 4) / 2;
    source_char_offset_ = (origin_lane_id_ % 2) * 2;

    lane_id_[0] = lane_id / 8;
    for(int i = 0; i < 3; i += 1){
      lane_id_[i + 1] = lane_id_[i] + 4;
    }

    for(int i = 0; i < 4; i += 1){
      lane_id = lane_id_[i];
      int lane_in_quad_pair = (lane_id & 7);

      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

      int k_group_idx_tmp = 0;

      // stride_: 24
      // Layout::kFactor: 2
      // Policy::LdsmShape::kContiguous: 1
      // Policy::LdsmShape::kStrided: 4
      // Shape::kStrided: 32
      // Shape::kContiguous: 32
      // InstructionShape::kStrided: 16
      // InstructionShape::kContiguous: 8
      // Policy::LdsmIterations::kContiguous: 1
      // Policy::LdsmIterations::kStrided: 1
      // Layout::PartitionShape::kContiguous: 4
      // Layout::PartitionShape::kStrided: 4

      // Super Matrix multiply kBlock = 32
      // Matrix multiply 1688 A/B
      // (Q stands for 1 8x128bit block).
      // Q0
      // Q1
      // Q2
      // Q3
      // Four blocks are next to each other in the strided dimension.
      partition_contiguous_idx = (lane_id % Layout::kFactor);
      access_contiguous_idx = (lane_in_quad_pair / Layout::kFactor);
      access_strided_idx = lane_id / Layout::kFactor;

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      byte_offset_[i] = (access_contiguous + access_strided * stride_) *
                    sizeof_bits<Element>::value * Layout::kElementsPerAccess / 8;

      if((origin_lane_id_ % 8) / 4 >= 1){
        for(int j = 0; j < 2; j += 1){
            if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
              int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                            ? 3
                            : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);

              if (((k_group_idx_tmp & mask) % 2) == 0)
                byte_offset_[i] ^= 1 * Policy::LdsmShape::kContiguous *
                                sizeof_bits<Element>::value *
                                Layout::kElementsPerAccess / 8;
              else if ((k_group_idx_tmp & mask) == 1)
                byte_offset_[i] ^= 3 * Policy::LdsmShape::kContiguous *
                                sizeof_bits<Element>::value *
                                Layout::kElementsPerAccess / 8;
              else if ((k_group_idx_tmp & mask) == 3)
                byte_offset_[i] ^= 7 * Policy::LdsmShape::kContiguous *
                                sizeof_bits<Element>::value *
                                Layout::kElementsPerAccess / 8;
            }

            k_group_idx_tmp++;
        }

      }
    }

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &add_tile_offset(
      TensorCoord const &tile_offset) {
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    for(int i = 0; i < 4; i += 1){
      byte_offset_[i] ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;
    }
    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor / 2 +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
        whole_tiles -= 1;
        k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) * 
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) * 
                        Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    k_group_idx_ += k_groups_delta;
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &operator++() {

    // Integer matrix multiply 16832 Interleaved-32
    //   NONE
    // Integer matrix multiply 16816 Interleaved-32 || Integer matrix multiply 16816 kblock=32

    // Integer matrix multiply 8816  Interleaved-32
    //   ^1 ^1
    // Matrix multiply 1684.TF32 kblock=16 || Integer matrix multiply 16816 kblock=64
    // Matrix multiply 1688 kblock=32 || Integer matrix multiply 8816 kblock=64
    //   ^1 ^3 ^1 ^3
    // Matrix multiply 1688 kblock=64
    //   ^1 ^3 ^1 ^7 ^1 ^3 ^1 ^7

    // Matrix multiply 16816 kblock=32 | 1688.TF32 kblock=16 || Integer matrix multiply 16832 kblock=64
    //   ^2 ^2
    // Matrix multiply 16816 kblock=64 | 1688.TF32 kblock=32 || Integer matrix multiply 16832 kblock=128
    //   ^2 ^6 ^2 ^6

    // Policy::kGroupsPerTile: 4
    // kPartitionsK: 1
    // Layout::kElementsPerAccess: 8
    // xor_val 0: 16
    // xor_val 1: 48
    if(k_group_idx_ == 0){
      source_int_offset_ += 2;
    }
    else if(k_group_idx_ == 1){
      source_int_offset_ -= 2;
      for(int i = 0; i < 4; i += 1){
        byte_offset_[i] ^= 1 * Policy::LdsmShape::kContiguous *
                          sizeof_bits<Element>::value *
                          Layout::kElementsPerAccess / 8;
      }
    }
    else if(k_group_idx_ == 2){
      source_int_offset_ += 2;
    }
    else if(k_group_idx_ == 3){
      source_int_offset_ -= 2;
      for(int i = 0; i < 4; i += 1){
        byte_offset_[i] ^= 1 * Policy::LdsmShape::kContiguous *
                          sizeof_bits<Element>::value *
                          Layout::kElementsPerAccess / 8;
      }
    }

    /*
    for(int i = 0; i < 4; i += 1){
      if(k_group_idx_ == 0){
        source_int_offset_ += 2;
      }
      else if(k_group_idx_ == 1){
        source_int_offset_ -= 2;
        byte_offset_[i] ^= 1 * Policy::LdsmShape::kContiguous *
                          sizeof_bits<Element>::value *
                          Layout::kElementsPerAccess / 8;
      }
      else if(k_group_idx_ == 2){
        source_int_offset_ += 2;
      }
      else if(k_group_idx_ == 3){
        source_int_offset_ -= 2;
        byte_offset_[i] ^= 1 * Policy::LdsmShape::kContiguous *
                          sizeof_bits<Element>::value *
                          Layout::kElementsPerAccess / 8;
      }
    }
    */
    

    k_group_idx_++;

    if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
      k_group_idx_ = 0;
      add_tile_offset({Policy::kGroupsPerTile, 0});
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(RealFragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      RealFragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    int8_t *fetch_ptr_byte = reinterpret_cast<int8_t *>(&frag);

    // stride_: 24
    AccessType *source_ptr = pointer_;

    for(int i = 0; i < 4; i += 1){
      char *source_byte_ptr = 
        reinterpret_cast<char *>(source_ptr) + byte_offset +
        byte_offset_[i];
      int load_int_tmp = *((int*)(source_byte_ptr) + source_int_offset_);
      int8_t *load_int_ptr = reinterpret_cast<int8_t *>(&load_int_tmp);
      //*(fetch_ptr_byte + i * 2) = *(load_int_ptr + source_char_offset_);
      //*(fetch_ptr_byte + i * 2 + 1) = *(load_int_ptr + source_char_offset_ + 1);

      frag[i * 2] = *(load_int_ptr + source_char_offset_);
      frag[i * 2 + 1] = *(load_int_ptr + source_char_offset_ + 1);
    }

    auto batch_id = blockIdx.z;
    auto head_id = blockIdx.y;
    auto warp_id = threadIdx.y;
    // printf("source_int_offset_: %d, source_char_offset_: %d, sizeof_bits<Element>::value: %d, Policy::LdsmShape::kCount: %d, batch_id: %d, head_id: %d, warp_id: %d, lane_id: %d, sb[0]: %f, sb[1]: %f, sb[2]: %f, sb[3]: %f, sb[4]: %f, sb[5]: %f, sb[6]: %f, sb[7]: %f\n", source_int_offset_, source_char_offset_, sizeof_bits<Element>::value, Policy::LdsmShape::kCount, batch_id, head_id, warp_id, origin_lane_id_, float(frag[0]), float(frag[1]), float(frag[2]), float(frag[3]), float(frag[4]), float(frag[5]), float(frag[6]), float(frag[7]));
    // printf("sizeof_bits<Element>::value: %d, Policy::LdsmShape::kCount: %d, batch_id: %d, head_id: %d, warp_id: %d, lane_id: %d, sb[0]: %f, sb[1]: %f, sb[2]: %f, sb[3]: %f, sb[4]: %f, sb[5]: %f, sb[6]: %f, sb[7]: %f, sx[0]: %f, sx[1]: %f, sx[2]: %f, sx[3]: %f, sx[4]: %f, sx[5]: %f, sx[6]: %f, sx[7]: %f\n", sizeof_bits<Element>::value, Policy::LdsmShape::kCount, batch_id, head_id, warp_id, lane_id_, float(*((int8_t *)source_byte_ptr+0)), float(*((int8_t *)source_byte_ptr+1)), float(*((int8_t *)source_byte_ptr+2)), float(*((int8_t *)source_byte_ptr+3)), float(*((int8_t *)source_byte_ptr+4)), float(*((int8_t *)source_byte_ptr+5)), float(*((int8_t *)source_byte_ptr+6)), float(*((int8_t *)source_byte_ptr+7)), float(*((int8_t *)fetch_ptr+0)), float(*((int8_t *)fetch_ptr+1)), float(*((int8_t *)fetch_ptr+2)), float(*((int8_t *)fetch_ptr+3)), float(*((int8_t *)fetch_ptr+4)), float(*((int8_t *)fetch_ptr+5)), float(*((int8_t *)fetch_ptr+6)), float(*((int8_t *)fetch_ptr+7)));
    // printf("sizeof_bits<Element>::value: %d, Policy::LdsmShape::kCount: %d, batch_id: %d, head_id: %d,lane_id: %d, sb[0]: %f, sb[1]: %f, sb[2]: %f, sb[3]: %f, sb[4]: %f, sb[5]: %f, sb[6]: %f, sb[7]: %f, sx[0]: %f, sx[1]: %f, sx[2]: %f, sx[3]: %f, sx[4]: %f, sx[5]: %f, sx[6]: %f, sx[7]: %f\n", sizeof_bits<Element>::value, Policy::LdsmShape::kCount, batch_id, head_id, lane_id_, float(*((Element *)source_byte_ptr+0)), float(*((Element *)source_byte_ptr+1)), float(*((Element *)source_byte_ptr+2)), float(*((Element *)source_byte_ptr+3)), float(*((Element *)source_byte_ptr+4)), float(*((Element *)source_byte_ptr+5)), float(*((Element *)source_byte_ptr+6)), float(*((Element *)source_byte_ptr+7)), float(*((Element *)fetch_ptr+0)), float(*((Element *)fetch_ptr+1)), float(*((Element *)fetch_ptr+2)), float(*((Element *)fetch_ptr+3)), float(*((Element *)fetch_ptr+4)), float(*((Element *)fetch_ptr+5)), float(*((Element *)fetch_ptr+6)), float(*((Element *)fetch_ptr+7)));
    // printf("\n", float(*((Element *)fetch_ptr+0)), float(*((Element *)fetch_ptr+1)), float(*((Element *)fetch_ptr+2)), float(*((Element *)fetch_ptr+3)));
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
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

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
    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator_Scale<
    Shape_, Operand_, Element_,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Crosswise may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// KBlock size
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
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
  using Base = MmaTensorOpMultiplicandTileIterator_Scale<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
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
  MmaTensorOpMultiplicandTileIterator_Scale() {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator_Scale &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(RealFragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    // TODO
    assert(0);
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
    assert(0);
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
        frag, {tile_offset.contiguous(), tile_offset.strided()}, byte_offset);
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

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
