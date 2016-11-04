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

#include "FPHeaderTable.h"

namespace cuda_fp_growth {

/// Get the sign of a given value (-1 if the value is negative, +1 if positive and 0 if it is zero)
__device__
inline int sgn( int x ) { return ( 0 < x ) - ( x < 0 ); }

__device__
void update_header_node( index_type item_idx, index_type node_idx, const NodeType& node_type, const size_type trans_count,
                         const Item* __restrict__ freq_items, const size_type ht_size, const size_type ia_size,
                         Item* items, size_type* counts, size_type* ia_sizes, index_type* ia_arrays, size_type* node_counts, NodeType* node_types )
{
    // we must use atomic operations to avoid data race among threads
    bool updating = false;
    const Item& current_item = freq_items[ item_idx ];
    do {
        // acquire lock by replacing the item with NullItem
        updating = ( atomicCAS( items + item_idx, current_item, NullItem ) == current_item );
        if ( updating ) {
            // entered critical section
            index_type pos = 0;
            size_type ia_array_size = ia_sizes[ item_idx ];
            size_type ia_offset = item_idx * ia_size;
            while ( pos < ia_array_size && ( ia_arrays[ ia_offset + pos ] != node_idx || node_types[ ia_offset + pos ] != node_type ) ) ++pos;
            if ( pos == ia_array_size ) {
                atomicExch( ia_arrays + ia_offset + pos, node_idx );
                atomicExch( node_counts + ia_offset + pos, trans_count );
                atomicExch( (cuda_int*) node_types + ia_offset + pos, (cuda_int) node_type );
                // memory fence to ensure any thread sees the increased ia_sizes also sees the changes we've put in
                __threadfence();
                atomicInc( ia_sizes + item_idx, ia_size );
            }
            else {
                atomicAdd( node_counts + ia_offset + pos, trans_count );
            }
            atomicAdd( counts + item_idx, trans_count );
            // memory fence to ensure any thread sees release of lock also sees our changes
            __threadfence();
            // put the item back to indicate release of lock
            atomicExch( items + item_idx, current_item );
            __threadfence();
        }
    } while ( !updating );
}

__device__
void link_inner_node( const InnerNode* __restrict__ inner_nodes, const index_type inner_idx, const size_type use_trans_count,
                      const Item* __restrict__ freq_items, const size_type ht_size, const size_type ia_size, const BitBlock* __restrict__ trans_map, const size_type blocks_per_trans,
                      Item* items, size_type* counts, size_type* ia_sizes, index_type* ia_arrays, size_type* node_counts, NodeType* node_types )
{
    assert( std::numeric_limits<BitBlock>::digits == 8 );
    const InnerNode* __restrict__ node = inner_nodes + inner_idx;
    const InnerNode* __restrict__ parent = inner_nodes + node->parent_idx;
    size_type parent_length = parent->prefix_length;
    // skip parent prefix, only scan remaining items
    for ( index_type item_idx = parent_length; item_idx < min( node->prefix_length, ht_size ); ++item_idx ) {
        // because all transactions covered by this inner node share same prefix, we use the first one to identify the items covered by this inner node
        index_type block_idx = node->range_start * blocks_per_trans + item_idx / 8;
        index_type offset = 7 - ( item_idx % 8 );
        BitBlock mask = 1u << offset;
        if ( ( trans_map[ block_idx ] & mask ) != 0 ) {
            // if trans_count is not specified, read it from the node itself
            size_type trans_count = use_trans_count == 0 ? node->trans_count : use_trans_count;
            update_header_node( item_idx, inner_idx, NodeType::InnerNode, trans_count, freq_items, ht_size, ia_size,
                                items, counts, ia_sizes, ia_arrays, node_counts, node_types );
        }
    }
}

__device__
void link_leaf_node( const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes, const index_type leaf_idx, const size_type use_trans_count,
                     const Item* __restrict__ freq_items, const size_type ht_size, const size_type ia_size, const BitBlock* __restrict__ trans_map, const size_type blocks_per_trans,
                     Item* items, size_type* counts, size_type* ia_sizes, index_type* ia_arrays, size_type* node_counts, NodeType* node_types )
{
    assert( std::numeric_limits<BitBlock>::digits == 8 );
    const LeafNode* node = leaf_nodes + leaf_idx;
    const InnerNode* parent = inner_nodes + node->parent_idx;
    size_type prefix_length = parent->prefix_length;
    // skip shared prefix, only scan remaining items
    for ( index_type item_idx = prefix_length; item_idx < ht_size; ++item_idx ) {
        index_type block_idx = leaf_idx * blocks_per_trans + item_idx / 8;
        index_type offset = 7 - ( item_idx % 8 );
        BitBlock mask = 1u << offset;
        if ( ( trans_map[ block_idx ] & mask ) != 0 ) {
            // if trans_count is not specified, read it from the node itself
            size_type trans_count = use_trans_count == 0 ? node->trans_count : use_trans_count;
            update_header_node( item_idx, leaf_idx, NodeType::LeafNode, trans_count, freq_items, ht_size, ia_size,
                                items, counts, ia_sizes, ia_arrays, node_counts, node_types );
        }
    }
}

__device__
void link_node_path( const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes,
                     index_type node_idx, NodeType node_type, const size_type node_count,
                     const Item* __restrict__ freq_items, const size_type ht_size, const size_type ia_size,
                     const BitBlock* __restrict__ trans_map, size_type blocks_per_trans,
                     Item* items, size_type* counts, size_type* ia_sizes, index_type* ia_arrays, size_type* node_counts, NodeType* node_types )
{
    while ( node_type == NodeType::LeafNode || node_idx != 0 ) {
        if ( node_type == NodeType::InnerNode ) {
            link_inner_node( inner_nodes, node_idx, node_count, freq_items, ht_size, ia_size,
                             trans_map, blocks_per_trans, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
            node_idx = inner_nodes[ node_idx ].parent_idx;
        }
        else {
            link_leaf_node( inner_nodes, leaf_nodes, node_idx, node_count, freq_items, ht_size, ia_size,
                            trans_map, blocks_per_trans, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
            node_idx = leaf_nodes[ node_idx ].parent_idx;
        }
        node_type = NodeType::InnerNode;
    }
}

__global__
void construct_initial_header_table( const Item* __restrict__ freq_items, const BitBlock* __restrict__ trans_map, size_type blocks_per_trans,
                                     const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes,
                                     const size_type ht_size, const size_type ia_size, cuda_uint* output )
{
    // declare dynamic shared memory buffer and fill with 0
    extern __shared__ cuda_uint buffer[];

    // prepare pointers to different parts of it
    Item* items = reinterpret_cast<Item*>( (size_type*) buffer + 2 );
    size_type* counts = reinterpret_cast<size_type*>( items + ht_size );
    size_type* ia_sizes = reinterpret_cast<size_type*>( counts + ht_size );
    index_type* ia_arrays = reinterpret_cast<index_type*>( ia_sizes + ht_size );
    size_type* node_counts = reinterpret_cast<size_type*>( ia_arrays + ht_size * ia_size );
    NodeType* node_types = reinterpret_cast<NodeType*>( node_counts + ht_size * ia_size );

    // initialize shared buffer
    if ( threadIdx.x == 0 ) {
        memset( buffer, 0, HTBufferSize( ht_size, ia_size ) );
        *( reinterpret_cast<size_type*>( buffer ) ) = ht_size;
        *( reinterpret_cast<size_type*>( buffer ) + 1 ) = ia_size;
        memcpy( items, freq_items, sizeof( Item ) * ht_size );
    }
    __syncthreads();

    const InnerNode* __restrict__ node = inner_nodes + threadIdx.x;

    if ( node->left_is_leaf ) {
        link_leaf_node( inner_nodes, leaf_nodes, node->left_idx, 0, freq_items, ht_size, ia_size,
                        trans_map, blocks_per_trans, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
    }
    else {
        link_inner_node( inner_nodes, node->left_idx, 0, freq_items, ht_size, ia_size,
                         trans_map, blocks_per_trans, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
    }

    if ( node->right_is_leaf ) {
        link_leaf_node( inner_nodes, leaf_nodes, node->right_idx, 0, freq_items, ht_size, ia_size,
                        trans_map, blocks_per_trans, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
    }
    else {
        link_inner_node( inner_nodes, node->right_idx, 0, freq_items, ht_size, ia_size,
                         trans_map, blocks_per_trans, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
    }

    __syncthreads();
    if ( threadIdx.x == 0 ) memcpy( output, buffer, HTBufferSize( ht_size, ia_size ) );
}

__global__
void construct_sub_header_table( const BitBlock* __restrict__ trans_map, const size_type blocks_per_trans,
                                 const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes,
                                 const cuda_uint* __restrict__ parent_ht, const index_type ht_idx, cuda_uint* output )
{
    // declare dynamic shared memory buffer and fill with 0
    extern __shared__ cuda_uint buffer[];

    size_type ht_size = *( reinterpret_cast<const size_type*>( parent_ht ) ), ia_size = *( reinterpret_cast<const size_type*>( parent_ht ) + 1 );

    // prepare pointers to different parts of it
    Item* items = reinterpret_cast<Item*>( (size_type*) buffer + 2 );
    size_type* counts = reinterpret_cast<size_type*>( items + ht_size );
    size_type* ia_sizes = reinterpret_cast<size_type*>( counts + ht_size );
    index_type* ia_arrays = reinterpret_cast<index_type*>( ia_sizes + ht_size );
    size_type* node_counts = reinterpret_cast<size_type*>( ia_arrays + ht_size * ia_size );
    NodeType* node_types = reinterpret_cast<NodeType*>( node_counts + ht_size * ia_size );

    // prepare pointers to parent header table
    const Item* __restrict__ parent_items = reinterpret_cast<const Item*>( (size_type*) parent_ht + 2 );
    const size_type* __restrict__ parent_counts = reinterpret_cast<const size_type*>( parent_items + ht_size );
    const size_type* __restrict__ parent_ia_sizes = reinterpret_cast<const size_type*>( parent_counts + ht_size );
    const index_type* __restrict__ parent_ia_arrays = reinterpret_cast<const index_type*>( parent_ia_sizes + ht_size );
    const size_type* __restrict__ parent_node_counts = reinterpret_cast<const size_type*>( parent_ia_arrays + ht_size * ia_size );
    const NodeType* __restrict__ parent_node_types = reinterpret_cast<const NodeType*>( parent_node_counts + ht_size * ia_size );

    if ( threadIdx.x == 0 ) {
        memset( buffer, 0, HTBufferSize( ht_size, ia_size ) );
        memcpy( buffer, parent_ht, sizeof( cuda_uint ) * ( 2 + ht_size ) );
    }
    __syncthreads();

    size_type offset = ht_idx * ia_size;
    index_type node_idx = parent_ia_arrays[ offset + threadIdx.x ];
    NodeType node_type = parent_node_types[ offset + threadIdx.x ];
    size_type node_count = parent_node_counts[ offset + threadIdx.x ];
    // walk up path and link nodes
    link_node_path( inner_nodes, leaf_nodes, node_idx, node_type, node_count, parent_items, ht_idx, ia_size, trans_map, blocks_per_trans,
                    items, counts, ia_sizes, ia_arrays, node_counts, node_types );

    __syncthreads();
    if ( threadIdx.x == 0 ) memcpy( output, buffer, HTBufferSize( ht_size, ia_size ) );
}

__host__
FPHeaderTable::FPHeaderTable( const FPTransMap& trans_map, const FPRadixTree& radix_tree, const size_type min_support )
    : _trans_map( trans_map.bitmap().data().get() ), _blocks_per_trans( trans_map.blocks_per_transaction() ),
      _inner_nodes( radix_tree.inner_nodes().data().get() ), _leaf_nodes( radix_tree.leaf_nodes().data().get() ),
      _ht_size( trans_map.frequent_items().size() ), _ia_size( trans_map.max_frequency() )
{
    // Construct header table from radix tree
    assert( _ht_size > 0 );
    assert( _ia_size > 0 );
    cudaCheck( cudaMalloc( &_data, HTBufferSize( _ht_size, _ia_size ) ) );
    construct_initial_header_table <<< 1, trans_map.size() - 1, HTBufferSize( _ht_size, _ia_size ) >>> (
        trans_map.frequent_items().data().get(), _trans_map, _blocks_per_trans,
        _inner_nodes, _leaf_nodes, _ht_size, _ia_size, _data
    );
    cudaDeviceSynchronize();
    cudaCheck( cudaMemcpy( &_ia_size, reinterpret_cast<size_type*>( _data ) + 1, sizeof( size_type ), cudaMemcpyDeviceToHost ) );

    // prepare pointers for easy access
    _items = reinterpret_cast<Item*>( (size_type*) _data + 2 );
    _counts = reinterpret_cast<size_type*>( _items + _ht_size );
    _ia_sizes = reinterpret_cast<size_type*>( _counts + _ht_size );
    _ia_arrays = reinterpret_cast<index_type*>( _ia_sizes + _ht_size );
    _node_counts = reinterpret_cast<size_type*>( _ia_arrays + _ht_size * _ia_size );
    _node_types = reinterpret_cast<NodeType*>( _node_counts + _ht_size * _ia_size );
}

__device__
FPHeaderTable::FPHeaderTable( const BitBlock* __restrict__ trans_map, const size_type blocks_per_trans,
                              const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes,
                              const cuda_uint* __restrict__ parent_ht, const size_type min_support, const index_type node_idx )
    : _trans_map( trans_map ), _blocks_per_trans( blocks_per_trans ), _inner_nodes( inner_nodes ), _leaf_nodes( leaf_nodes )
{
    // Allocate memory for sub header table
    size_type ht_size = *( reinterpret_cast<const size_type*>( parent_ht ) ), ia_size = *( reinterpret_cast<const size_type*>( parent_ht ) + 1 );
    _data = new cuda_uint[ HTBufferSize( ht_size, ia_size ) / sizeof( cuda_uint ) ];

    // Construct sub header table
    const size_type* __restrict__ parent_ia_sizes = reinterpret_cast<const size_type*>( parent_ht ) + 2 + 2 * ht_size;
    size_type node_ia_size = parent_ia_sizes[ node_idx ];
    construct_sub_header_table <<< 1, node_ia_size, HTBufferSize( ht_size, ia_size ) >>>(
        trans_map, blocks_per_trans, inner_nodes, leaf_nodes, parent_ht, node_idx, _data
    );
    cudaDeviceSynchronize();
    _ht_size = *( reinterpret_cast<const size_type*>( _data ) );
    _ia_size = *( reinterpret_cast<const size_type*>( _data ) + 1 );

    // prepare pointers for easy access
    _items = reinterpret_cast<Item*>( (size_type*) _data + 2 );
    _counts = reinterpret_cast<size_type*>( _items + _ht_size );
    _ia_sizes = reinterpret_cast<size_type*>( _counts + _ht_size );
    _ia_arrays = reinterpret_cast<index_type*>( _ia_sizes + _ht_size );
    _node_counts = reinterpret_cast<size_type*>( _ia_arrays + _ht_size * _ia_size );
    _node_types = reinterpret_cast<NodeType*>( _node_counts + _ht_size * _ia_size );
}

__host__ __device__
FPHeaderTable::~FPHeaderTable()
{
#ifndef __CUDA_ARCH__
    cudaCheck( cudaFree( _data ) );
#else
    delete[] _data;
#endif // __CUDA_ARCH__
}

#ifdef UNIT_TEST

Items FPHeaderTable::get_items() const
{
    thrust::device_vector<Item> items( _items, _items + _ht_size );
    return Items( items.begin(), items.end() );
}

Sizes FPHeaderTable::get_counts() const
{
    thrust::device_vector<size_type> counts( _counts, _counts + _ht_size );
    return Sizes( counts.begin(), counts.end() );
}

Sizes FPHeaderTable::get_ia_sizes() const
{
    thrust::device_vector<size_type> counts( _ia_sizes, _ia_sizes + _ht_size );
    return Sizes( counts.begin(), counts.end() );
}

Indices FPHeaderTable::get_ia_array( index_type idx ) const
{
    const index_type* ptr = _ia_arrays + idx * _ia_size;
    size_type ia_size;
    cudaCheck( cudaMemcpy( &ia_size, _ia_sizes + idx, sizeof( size_type ), cudaMemcpyDeviceToHost ) );
    thrust::device_vector<index_type> indices( ptr , ptr + ia_size );
    return Indices( indices.begin(), indices.end() );
}

Sizes FPHeaderTable::get_node_counts( index_type idx ) const
{
    const size_type* ptr = _node_counts + idx * _ia_size;
    size_type ia_size;
    cudaCheck( cudaMemcpy( &ia_size, _ia_sizes + idx, sizeof( size_type ), cudaMemcpyDeviceToHost ) );
    thrust::device_vector<size_type> counts( ptr, ptr + ia_size );
    return Sizes( counts.begin(), counts.end() );
}

NodeTypes FPHeaderTable::get_node_types( index_type idx ) const
{
    const NodeType* ptr = _node_types + idx * _ia_size;
    size_type ia_size;
    cudaCheck( cudaMemcpy( &ia_size, _ia_sizes + idx, sizeof( size_type ), cudaMemcpyDeviceToHost ) );
    thrust::device_vector<NodeType> flags( ptr, ptr + ia_size );
    return NodeTypes( flags.begin(), flags.end() );
}

#endif // UNIT_TEST

}
