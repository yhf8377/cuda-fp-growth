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

#ifndef CUDA_FP_GROWTH_FPGROWTH_H
#define CUDA_FP_GROWTH_FPGROWTH_H

#include "FPTransMap.h"
#include "FPRadixTree.h"
#include "FPHeaderTable.h"

namespace cuda_fp_growth {

class FPGrowth
{
    public:
        FPGrowth( const FPTransMap& trans_map, const FPRadixTree& radix_tree, const FPHeaderTable& ht, size_type min_support );
        virtual ~FPGrowth() = default;

        /** \brief Mine all frequent patterns
         *
         * Before calling this method, an output buffer must be allocated on the host. Its pointer and size are passed to this method.
         * This method will mine all frequent patterns that satisfy the minimum support, and store these patterns into the output buffer.
         * Once the output buffer is filled or all frequent patterns are found, this method exits.
         *
         * The patterns will be saved to the buffer in a streaming format: [size, support, item_0, item_1, ..., item_n, size, ...] where:
         *     - size is the total size of this pattern (including the `size` field). `size` equals to sizeof( cuda_uint ) * ( n + 2 ).
         *     - support is the support count of this pattern
         *     - item_0 to item_n are n items that form the pattern
         *
         * \param buffer a pointer to the output buffer
         * \param buffer_size the size of the output buffer. Upon method return it stores the actual size that were used to store the patterns.
         * \return void
         *
         */
        void mine_frequent_patterns( cuda_uint* buffer, size_type& buffer_size ) const;

    private:
        const FPTransMap& _trans_map;
        const FPRadixTree& _radix_tree;
        const FPHeaderTable& _ht;
        const size_type _min_support;
};

}

#endif // CUDA_FP_GROWTH_FPGROWTH_H
