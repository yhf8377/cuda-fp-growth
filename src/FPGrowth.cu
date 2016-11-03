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

__global__
void parallel_mine_tree( const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes, const cuda_uint* __restrict__ ht_data,
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
            parallel_mine_tree <<< 1, sub_ht.size() >>> ( inner_nodes, leaf_nodes, sub_ht.data(), trans_map, blocks_per_trans, new_prefix, prefix_length + 1, min_support );
            cudaDeviceSynchronize();
        }
    }
    __syncthreads();
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
        parallel_mine_tree <<< 1, _ht.size() >>> (
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

}
