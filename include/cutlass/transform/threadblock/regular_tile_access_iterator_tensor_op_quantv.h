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
    \brief Templates implementing computing the addresses of storing of tiles
   from pitch-linear rank=2 tensors.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {
template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap,
          int Alignment =
              sizeof_bits<Element>::value* ThreadMap::kElementsPerAccess / 8>
class RegularTileAccessIterator_QuantV;


////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIterator_QuantV<
    Shape_, Element_,
    layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                          Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in
    /// length.
    static int const kAccessSizeInBits = 128;

    static_assert(sizeof_bits<Element_>::value *
                          ThreadMap::kElementsPerAccess ==
                      kAccessSizeInBits,
                  "This iterator requires a policy whose access size is 128bs");

    ///< Number of pointers
    static int const kPointerCount =
        (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);
  };

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType *pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator_QuantV(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : stride_(ref.stride(0) / Layout::kElementsPerAccess),
        byte_offset_(0) {
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);
    
    // ThreadMap::Iterations::kContiguous: 1
    // ThreadMap::Iterations::kStrided: 1
    // ThreadMap::Detail::WarpThreadArrangement::kStrided: 8

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // This is the offset of a thread within a threadblock tile for a specific
      // pointer (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};

      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(
          ref.data() + ref.offset(thread_offset_in_threadblock_tile));
      // printf("kPointerCount: %d, thread_id: %d, thread_offset_in_threadblock_tile.contiguous(): %d, thread_offset_in_threadblock_tile.strided(): %d\n", i, thread_id, thread_offset_in_threadblock_tile.contiguous(), thread_offset_in_threadblock_tile.strided());
    }

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    AccessType *access_ptr = pointer_[iteration_strided_ & 1];
    int stride_idx = (iteration_strided_ & ~1);

    // printf("ThreadMap::Delta::kContiguous: %d, ThreadMap::kElementsPerAccess: %d\n", ThreadMap::Delta::kContiguous, ThreadMap::kElementsPerAccess);

    int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
                        // need to multiply 2 since contiguous stride is 128
                        iteration_contiguous_ * ThreadMap::Delta::kContiguous * 2 /
                            ThreadMap::kElementsPerAccess;

    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator_QuantV &operator++() {
    // printf("ThreadMap::Iterations::kContiguous: %d, ThreadMap::Iterations::kStrided: %d\n", ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided);
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    // Enter here only if (iteration_strided_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    iteration_strided_ = 0;

    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator_QuantV operator++(int) {
    RegularTileAccessIterator_QuantV prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * Shape::kContiguous +
                       coord.strided() * Shape::kStrided * stride_ *
                           Layout::kElementsPerAccess);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIterator_QuantV<
    Shape_, Element_,
    layout::RowMajorTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                  Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for row-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileAccessIterator_QuantV<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator_QuantV(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : iterator_({ref.data(), ref.stride()}, thread_id) {
        // printf("Shape::kContiguous: %d, Shape::kStrided: %d, kElementsPerAccess: %d, Detail::WarpThreadArrangement::kContiguous: %d, Detail::WarpThreadArrangement::kStrided: %d, ShapeInAccesses::kContiguous: %d, ShapeInAccesses::kStrided: %d, Detail::WarpAccessIterations::kContiguous: %d, Detail::WarpAccessIterations::kStrided: %d, Detail::WarpArrangement::kContiguous: %d, Detail::WarpArrangement::kStrided: %d, Iterations::kContiguous: %d, Iterations::kStrided: %d, Delta::kContiguous: %d, Delta::kStrided: %d\n", ThreadMap::Shape::kContiguous, ThreadMap::Shape::kStrided, ThreadMap::kElementsPerAccess, ThreadMap::Detail::WarpThreadArrangement::kContiguous, ThreadMap::Detail::WarpThreadArrangement::kStrided, ThreadMap::Detail::ShapeInAccesses::kContiguous, ThreadMap::Detail::ShapeInAccesses::kStrided, ThreadMap::Detail::WarpAccessIterations::kContiguous, ThreadMap::Detail::WarpAccessIterations::kStrided, ThreadMap::Detail::WarpArrangement::kContiguous, ThreadMap::Detail::WarpArrangement::kStrided, ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided, ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided);
      }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator_QuantV &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator_QuantV operator++(int) {
    RegularTileAccessIterator_QuantV prev(*this);
    ++iterator_;

    return prev;
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
