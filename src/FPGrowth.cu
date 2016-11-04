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

#include "FPGrowth.h"

namespace cuda_fp_growth {

__device__ cuda_uint* _output_buffer;
__device__ size_type _output_buffer_size;
__device__ size_type _output_buffer_pos;

__device__
const Item* save_pattern( const Item* prefix, const size_type prefix_length, const Item& item, size_type support )
{
    size_type pattern_size = sizeof( cuda_uint ) * ( 3 + prefix_length );
    size_type pos = atomicAdd( &_output_buffer_pos, pattern_size );
    if ( pos + pattern_size <= _output_buffer_size ) {
        cuda_uint* ptr = reinterpret_cast<cuda_uint*>( (char*) _output_buffer + pos );
        ptr[ 0 ] = pattern_size;
        ptr[ 1 ] = support;
        memcpy( ptr + 2, prefix, sizeof( Item ) * prefix_length );
        *( ptr + 2 + prefix_length ) = item;
        return reinterpret_cast<const Item*>( ptr + 2 );
    }
    else {
        printf( "Insufficient buffer size!\n" );
        return nullptr;
    }
}

__device__
const Item* save_rule( const Item* prefix, const size_type prefix_length, const Item& item, size_type support, cuda_real confidence )
{
    static_assert( sizeof( cuda_uint ) == sizeof( cuda_real ), "size of cuda_uint not equal to cuda_real, so we can not save the cuda_real in a cuda_uint buffer" );
    size_type pattern_size = sizeof( cuda_uint ) * ( 4 + prefix_length );
    size_type pos = atomicAdd( &_output_buffer_pos, pattern_size );
    if ( pos + pattern_size <= _output_buffer_size ) {
        cuda_uint* ptr = reinterpret_cast<cuda_uint*>( (char*) _output_buffer + pos );
        ptr[ 0 ] = pattern_size;
        ptr[ 1 ] = support;
        ptr[ 2 ] = *( reinterpret_cast<cuda_uint*>( &confidence ) );
        memcpy( ptr + 3, prefix, sizeof( Item ) * prefix_length );
        *( ptr + 3 + prefix_length ) = item;
        return reinterpret_cast<const Item*>( ptr + 3 );
    }
    else {
        printf( "Insufficient buffer size!\n" );
        return nullptr;
    }
}

__global__
void parallel_mine_patterns( const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes, const cuda_uint* __restrict__ ht_data,
                            const BitBlock* __restrict__ trans_map, const size_type blocks_per_trans,
                            const Item* __restrict__ prefix, const size_type prefix_length, const size_type min_support )
{
    size_type ht_size = ht_data[0];
    const Item* items = reinterpret_cast<const Item*>( ht_data + 2 );
    const size_type* counts = reinterpret_cast<const size_type*>( items + ht_size );

    index_type idx = threadIdx.x;
    if ( counts[ idx ] >= min_support ) {
        const Item* new_prefix = save_pattern( prefix, prefix_length, items[ idx ], counts[ idx ] );
        if ( new_prefix != nullptr ) {
            FPHeaderTable sub_ht( trans_map, blocks_per_trans, inner_nodes, leaf_nodes, ht_data, min_support, idx );
            const cuda_uint* sub_ht_data = sub_ht.data();
            parallel_mine_patterns <<< 1, sub_ht.size() >>> ( inner_nodes, leaf_nodes, sub_ht_data, trans_map, blocks_per_trans, new_prefix, prefix_length + 1, min_support );
            cudaDeviceSynchronize();
        }
    }
}

__global__
void parallel_mine_rules( const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes, const cuda_uint* __restrict__ ht_data,
                          const BitBlock* __restrict__ trans_map, const size_type blocks_per_trans,
                          const Item* __restrict__ prefix, const size_type prefix_length, const index_type rhs_idx, const size_type min_support, cuda_real min_confidence )
{
    size_type ht_size = ht_data[0];
    const Item* items = reinterpret_cast<const Item*>( ht_data + 2 );
    const size_type* counts = reinterpret_cast<const size_type*>( items + ht_size );

    index_type idx = threadIdx.x;
    size_type lhs_count = counts[ idx ];
    if ( lhs_count >= min_support ) {
        // when count(A) satisfies count(A) >= min_support, we check the confidence by constructing its sub header table to get count(AB)
        FPHeaderTable sub_ht( trans_map, blocks_per_trans, inner_nodes, leaf_nodes, ht_data, min_support, idx );
        const cuda_uint* sub_ht_data = sub_ht.data();
        size_type sub_ht_size = sub_ht_data[ 0 ];
        const size_type* sub_ht_counts = sub_ht_data + 2 + sub_ht_size;

        size_type rhs_count = sub_ht_counts[ rhs_idx ];
        if ( rhs_count >= min_support * min_confidence ) {
            cuda_real confidence = (cuda_real) rhs_count / lhs_count;
            const Item* new_prefix = save_rule( prefix, prefix_length, items[ idx ], lhs_count, confidence );
            if ( new_prefix != nullptr ) {
                parallel_mine_rules <<< 1, sub_ht.size() >>> ( inner_nodes, leaf_nodes, sub_ht.data(), trans_map, blocks_per_trans, new_prefix, prefix_length + 1, rhs_idx, min_support, min_confidence );
            }
        }

        // wait until all child kernels has finished using sub_ht
        cudaDeviceSynchronize();
    }
}

FPGrowth::FPGrowth( const FPTransMap& trans_map, const FPRadixTree& radix_tree, const FPHeaderTable& ht, size_type min_support )
    : _trans_map( trans_map ), _radix_tree( radix_tree ), _ht( ht ), _min_support( min_support )
{
}

void FPGrowth::mine_frequent_patterns( cuda_uint* buffer, size_type& buffer_size ) const
{
    cuda_uint* output;
    cudaCheck( cudaMalloc( &output, buffer_size ) );
    try {
        // initialzie buffer on device
        size_type pos = 0;
        cudaCheck( cudaMemcpyToSymbol( _output_buffer, &output, sizeof( cuda_uint* ) ) );
        cudaCheck( cudaMemset( output, 0, buffer_size ) );
        cudaCheck( cudaMemcpyToSymbol( _output_buffer_size, &buffer_size, sizeof( size_type ) ) )
        cudaCheck( cudaMemcpyToSymbol( _output_buffer_pos, &pos, sizeof( size_type ) ) );

        // mine frequent patterns
        parallel_mine_patterns <<< 1, _ht.size() >>> (
            _radix_tree.inner_nodes().data().get(), _radix_tree.leaf_nodes().data().get(), _ht.data(),
            _trans_map.bitmap().data().get(), _trans_map.blocks_per_transaction(), nullptr, 0, _min_support
        );
        cudaDeviceSynchronize();

        // copy results
        cudaCheck( cudaMemcpyFromSymbol( &buffer_size, _output_buffer_pos, sizeof( size_type ) ) );
        cudaCheck( cudaMemcpy( buffer, output, buffer_size, cudaMemcpyDeviceToHost ) );
    }
    catch ( const std::exception& e ) {
#ifndef NDEBUG
        std::cout << __FILE__ << "#" << __LINE__ << ": " << e.what() << std::endl;
#endif // NDEBUG
    }
    // release buffer on device
    cudaCheck( cudaFree( output ) );
}

void FPGrowth::mine_association_rules( cuda_real min_confidence, cuda_uint* buffer, size_type& buffer_size ) const
{
    const DItems& freq_items = _trans_map.frequent_items();
    index_type rhs_idx = thrust::distance( freq_items.begin(), thrust::find( freq_items.begin(), freq_items.end(), _trans_map.rhs() ) );
    if ( rhs_idx >= freq_items.size() ) {
        buffer_size = 0;
        return;
    }

    cuda_uint* output;
    cudaCheck( cudaMalloc( &output, buffer_size ) );
    try {
        // initialzie buffer on device
        size_type pos = 0;
        cudaCheck( cudaMemcpyToSymbol( _output_buffer, &output, sizeof( cuda_uint* ) ) );
        cudaCheck( cudaMemset( output, 0, buffer_size ) );
        cudaCheck( cudaMemcpyToSymbol( _output_buffer_size, &buffer_size, sizeof( size_type ) ) )
        cudaCheck( cudaMemcpyToSymbol( _output_buffer_pos, &pos, sizeof( size_type ) ) );

        // mine frequent patterns
        parallel_mine_rules <<< 1, _ht.size() >>> (
            _radix_tree.inner_nodes().data().get(), _radix_tree.leaf_nodes().data().get(), _ht.data(),
            _trans_map.bitmap().data().get(), _trans_map.blocks_per_transaction(), nullptr, 0, rhs_idx, _min_support, min_confidence
        );
        cudaDeviceSynchronize();

        // copy results
        cudaCheck( cudaMemcpyFromSymbol( &buffer_size, _output_buffer_pos, sizeof( size_type ) ) );
        cudaCheck( cudaMemcpy( buffer, output, buffer_size, cudaMemcpyDeviceToHost ) );

        // remove rules with low confidence
        index_type input_pos = 0, output_pos = 0;
        while ( input_pos < buffer_size ) {
            index_type input_idx = input_pos / sizeof( cuda_uint );
            size_type length = buffer[ input_idx ];
            cuda_real confidence = *( reinterpret_cast<cuda_real*>( buffer + input_idx + 2 ) );
            if ( confidence >= min_confidence ) {
                if ( input_pos > output_pos ) memcpy( (char*) buffer + output_pos, (char*) buffer + input_pos, length );
                output_pos += length;
            }
            input_pos += length;
        }
        buffer_size = output_pos;
    }
    catch ( const std::exception& e ) {
#ifndef NDEBUG
        std::cout << __FILE__ << "#" << __LINE__ << ": " << e.what() << std::endl;
#endif // NDEBUG
    }
    // release buffer on device
    cudaCheck( cudaFree( output ) );
}

}
