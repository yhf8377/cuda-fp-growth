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

#ifndef CUDA_FP_GROWTH_FPHEADERTABLE_H
#define CUDA_FP_GROWTH_FPHEADERTABLE_H

#include "FPTransMap.h"
#include "FPRadixTree.h"

namespace cuda_fp_growth {

enum class NodeType : cuda_uint { InnerNode = 0, LeafNode = 1 };
using NodeTypes = std::vector<NodeType>;

#define HTBufferSize( ht_size, ia_size ) ( 2 * sizeof( size_type) + ( sizeof( Item ) + sizeof( size_type ) + sizeof( size_type ) + ( sizeof( index_type ) + sizeof( size_type ) + sizeof( NodeType ) ) * ia_size ) * ht_size )

class FPHeaderTable
{
    public:
        /** \brief Constructs initial header table from transaction map and radix tree
         *
         * \param trans_map an FPTransMap object
         * \param radix_tree an FPRadixTree object
         *
         */
        __host__
        FPHeaderTable( const FPTransMap& trans_map, const FPRadixTree& radix_tree, const size_type min_support );

        /** \brief Constructs sub header table for a node in parent header table
         *
         * \param parent_ht a pointer to the parent header table data in the constant cache
         * \param ht_size total number of header nodes in parent header table
         * \param ia_size size of the index array in the parent header table
         * \param min_suppport the minimum support required
         * \param node_idx the index of the header node to construct sub header table with
         *
         */
        __device__
        FPHeaderTable( const BitBlock* __restrict__ trans_map, const size_type blocks_per_trans,
                       const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes,
                       const cuda_uint* __restrict__ parent_ht, const size_type min_support, const index_type node_idx );

        __host__ __device__
        ~FPHeaderTable();

        __host__ __device__
        inline size_type size() const { return _ht_size; }

        __host__ __device__
        inline size_type ia_size() const { return _ia_size; }

        __host__ __device__
        inline const cuda_uint* data() const { return _data; }

#ifdef UNIT_TEST
        Items get_items() const;

        Sizes get_counts() const;

        Sizes get_ia_sizes() const;

        Indices get_ia_array( index_type idx ) const;

        Sizes get_node_counts( index_type idx ) const;

        NodeTypes get_node_types( index_type idx ) const;
#endif // UNIT_TEST

    private:
        const BitBlock* __restrict__ _trans_map;
        const InnerNode* __restrict__ _inner_nodes;
        const LeafNode* __restrict__ _leaf_nodes;
        const size_type _blocks_per_trans;
        cuda_uint* _data;
        size_type _ht_size, _ia_size;

        Item* _items;
        size_type* _counts;
        size_type* _ia_sizes;
        index_type* _ia_arrays;
        size_type* _node_counts;
        NodeType* _node_types;
};

}

#endif // CUDA_FP_GROWTH_FPHEADERTABLE_H
