/*
   Copyright 2016 Frank Ye

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#ifndef CUDA_FP_GROWTH_FPRADIXTREE_H
#define CUDA_FP_GROWTH_FPRADIXTREE_H

#include "FPTransMap.h"

namespace cuda_fp_growth {

struct InnerNode
{
    cuda_uint left_is_leaf, right_is_leaf;
    cuda_uint range_start, range_end, left_idx, right_idx, parent_idx;
    cuda_uint prefix_length, trans_count;
};
using InnerNodes = std::vector<InnerNode>;

struct LeafNode
{
    cuda_uint parent_idx;
    cuda_uint trans_count;
};
using LeafNodes = std::vector<LeafNode>;

using DInnerNodes = thrust::device_vector<InnerNode>;
using DLeafNodes = thrust::device_vector<LeafNode>;

class FPRadixTree
{
    public:
        /** \brief Constructs a binary radix tree from a transaction bitmap
         *
         * \param trans_map an FPTransactionMap object
         */
        FPRadixTree( const FPTransMap& trans_map );

        inline const DInnerNodes& inner_nodes() const { return _inner_nodes; }

        inline const DLeafNodes& leaf_nodes() const { return _leaf_nodes; }

    private:
        DInnerNodes _inner_nodes;
        DLeafNodes _leaf_nodes;
};

}

#endif // CUDA_FP_GROWTH_FPRADIXTREE_H
