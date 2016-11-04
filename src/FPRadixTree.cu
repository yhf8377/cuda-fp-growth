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

#include "FPRadixTree.h"

namespace cuda_fp_growth {

/// Get the sign of a given value (-1 if the value is negative, +1 if positive and 0 if it is zero)
__device__
inline int sgn( int x ) { return ( 0 < x ) - ( x < 0 ); }

/// Counts the length of the shared prefix between two transactions
__device__
int pfx( int i, int j, const BitBlock* __restrict__ trans_map, size_type n_trans, size_type blocks_per_trans )
{
    assert( std::numeric_limits<int>::digits == 31 );
    if ( i >= 0 && i < n_trans && j >= 0 && j < n_trans ) {
        int length = 0;
        for ( int k = 0; k < blocks_per_trans; ++k ) {
            BitBlock value_i = *( trans_map + i * blocks_per_trans + k );
            BitBlock value_j = *( trans_map + j * blocks_per_trans + k );
            BitBlock xor_value = value_i ^ value_j;
            if ( xor_value != 0 ) {
                length += ( __clz( xor_value ) - 24 );
                break;
            }
            else length += 8;
        }
        return length;
    }
    else return -1;
}

/// Counts total number of transactions covered by a transaction bitmap range
__device__
size_type count_transactions( index_type i, index_type j, const size_type* __restrict__ trans_counts )
{
    size_type n = 0;
    for ( index_type pos = i; pos <= j; ++pos ) n += *( trans_counts + pos );
    return n;
}

// Each inner node has 9x cuda_uint elements and each leaf node has 2x
#define InnerNodesSize( n_trans ) ( sizeof( cuda_uint ) * 9 * ( n_trans - 1 ) )
#define LeafNodesSize( n_trans ) ( sizeof( cuda_uint ) * 2 * n_trans )
#define RadixTreeBufferSize( n_trans ) ( sizeof( cuda_uint ) * ( 9 * ( n_trans - 1 ) + 2 * n_trans ) )

__global__
void construct_radix_tree( const BitBlock* __restrict__ trans_map, const size_type* __restrict__ trans_counts,
                           size_type n_trans, size_type blocks_per_trans, InnerNode* inner_nodes, LeafNode* leaf_nodes )
{
    extern __shared__ cuda_uint buffer[];
    if ( threadIdx.x == 0 ) memset( buffer, 0, RadixTreeBufferSize( n_trans ) );
    __syncthreads();

    InnerNode* _inner_nodes = reinterpret_cast<InnerNode*>( buffer );
    LeafNode* _leaf_nodes = reinterpret_cast<LeafNode*>( (char*)buffer + InnerNodesSize( n_trans ) );

    index_type i = threadIdx.x;

    // determine direction (+1 or -1)
    int d = sgn( pfx( i, i+1, trans_map, n_trans, blocks_per_trans ) - pfx( i, i-1, trans_map, n_trans, blocks_per_trans ) );

    // find upper-bound
    int min_pfx = pfx( i, i-d, trans_map, n_trans, blocks_per_trans );
    int l_max = 2;
    while ( pfx( i, i + l_max * d, trans_map, n_trans, blocks_per_trans ) > min_pfx ) l_max *= 2;

    // find the other end
    int l = 0;
    for ( int t = l_max / 2; t >= 1; t /= 2 ) {
        if ( pfx( i, i + ( l + t ) * d, trans_map, n_trans, blocks_per_trans ) > min_pfx ) l += t;
    }
    index_type j = i + l * d;

    // find split position
    int node_pfx = pfx( i, j, trans_map, n_trans, blocks_per_trans );
    int s = 0;
    for ( int t = l / 2; t >= 1; t /= 2 ) {
        if ( pfx( i, i + ( s + t ) * d, trans_map, n_trans, blocks_per_trans ) > node_pfx ) s += t;
    }
    if ( pfx( i, i + ( s + 1 ) * d, trans_map, n_trans, blocks_per_trans ) > node_pfx ) s += 1;
    index_type split = i + s * d + min( d, 0 );

    InnerNode* node = _inner_nodes + i;
    node->range_start = min( i, j );
    node->range_end = max( i, j );
    node->prefix_length = node_pfx;
    node->trans_count = count_transactions( node->range_start, node->range_end, trans_counts );
    // link left child
    node->left_idx = split;
    if ( min( i, j ) == split ) {
        node->left_is_leaf = true;
        ( _leaf_nodes + split )->parent_idx = i;
        ( _leaf_nodes + split )->trans_count = *( trans_counts + split );
    }
    else {
        node->left_is_leaf = false;
        ( _inner_nodes + split )->parent_idx = i;
    }
    // link right child
    node->right_idx = split + 1;
    if ( max( i, j ) == split + 1 ) {
        node->right_is_leaf = true;
        ( _leaf_nodes + split + 1 )->parent_idx = i;
        ( _leaf_nodes + split + 1 )->trans_count = *( trans_counts + split + 1 );
    }
    else {
        node->right_is_leaf = false;
        ( _inner_nodes + split + 1 )->parent_idx = i;
    }

    // copy results to output
    __syncthreads();
    if ( threadIdx.x == 0 ) {
        memcpy( inner_nodes, _inner_nodes, InnerNodesSize( n_trans ) );
        memcpy( leaf_nodes, _leaf_nodes, LeafNodesSize( n_trans ) );
    }
}

FPRadixTree::FPRadixTree( const FPTransMap& trans_map )
    : _inner_nodes( DInnerNodes( trans_map.size() - 1 ) ), _leaf_nodes( DLeafNodes( trans_map.size() ) )
{
    size_type n_trans = trans_map.size(), blocks_per_trans = trans_map.blocks_per_transaction();
    construct_radix_tree <<< 1, n_trans - 1, RadixTreeBufferSize( n_trans ) >>>(
        trans_map.bitmap().data().get(), trans_map.transaction_counts().data().get(), n_trans, blocks_per_trans,
        _inner_nodes.data().get(), _leaf_nodes.data().get()
    );
}

}
