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

#include "catch.hpp"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "FPHeaderTable.h"

namespace cuda_fp_growth {

__global__
void new_header_table( const BitBlock* __restrict__ trans_map, size_type blocks_per_trans,
                       const InnerNode* __restrict__ inner_nodes, const LeafNode* __restrict__ leaf_nodes,
                       const cuda_uint* __restrict__ parent_ht, size_type min_support, index_type node_idx, cuda_uint* output )
{
    FPHeaderTable sub_ht( trans_map, blocks_per_trans, inner_nodes, leaf_nodes, parent_ht, min_support, node_idx );
    cudaDeviceSynchronize();
    size_type ht_size = sub_ht.size(), ia_size = sub_ht.ia_size();
    memcpy( output, sub_ht.data(), HTBufferSize( ht_size, ia_size ) );
}

void test_sub_header_table( const FPTransMap& trans_map, const FPRadixTree& radix_tree, const FPHeaderTable& header_table, size_type min_support,
                            index_type node_idx, Items& items, Sizes& counts, Sizes& ia_sizes, Indices& ia_arrays, Sizes& node_counts, NodeTypes& node_types )
{
    thrust::device_vector<cuda_uint> output( HTBufferSize( header_table.size(), header_table.ia_size() ) / sizeof( cuda_uint ), 0 );
    cuda_uint* _output = output.data().get();
    new_header_table <<< 1, 1 >>>( trans_map.bitmap().data().get(), trans_map.blocks_per_transaction(),
                                   radix_tree.inner_nodes().data().get(), radix_tree.leaf_nodes().data().get(),
                                   header_table.data(), min_support, node_idx,  _output );
    cudaDeviceSynchronize();

    std::vector<cuda_uint> data( output.begin(), output.end() );
    size_type ht_size = data[ 0 ], ia_size = data[ 1 ];
    auto current = data.begin() + 2;

    items.clear();
    items.insert( items.end(), current, current + ht_size );
    current += ht_size;

    counts.clear();
    counts.insert( counts.end(), current, current + ht_size );
    current += ht_size;

    ia_sizes.clear();
    ia_sizes.insert( ia_sizes.end(), current, current + ht_size );
    current += ht_size;

    ia_arrays.clear();
    ia_arrays.insert( ia_arrays.end(), current, current + ht_size * ia_size );
    current += ht_size * ia_size;

    node_counts.clear();
    node_counts.insert( node_counts.end(), current, current + ht_size * ia_size );
    current += ht_size * ia_size;

    node_types.resize( ht_size * ia_size );
    std::transform( current, current + ht_size * ia_size, node_types.begin(), []( cuda_uint value ) { return static_cast<NodeType>( value ); } );
}

void sort_results( const Sizes& ia_sizes, Indices& ia_arrays, Sizes& node_counts, NodeTypes& node_types )
{
    for ( index_type i = 0; i < ia_sizes.size(); ++i ) {
        size_type ia_size = ia_sizes[ i ];
        size_type begin_pos = i * ia_size;

        std::vector<index_type> order( ia_size );
        std::iota( order.begin(), order.end(), begin_pos );
        std::sort( order.begin(), order.end(), [&]( index_type idx_a, index_type idx_b ) {
            NodeType type_a = node_types[ idx_a ], type_b = node_types[ idx_b ];
            index_type ia_a = ia_arrays[ idx_a ], ia_b = ia_arrays[ idx_b ];
            return ( type_a < type_b ) || ( type_a == type_b && ia_a < ia_b );
        } );

        Indices ordered_ia_arrays( ia_size );
        Sizes ordered_node_counts( ia_size );
        NodeTypes ordered_node_types( ia_size );
        for ( index_type i = 0; i < order.size(); ++i ) {
            ordered_ia_arrays[ i ] = ia_arrays[ order[ i ] ];
            ordered_node_counts[ i ] = node_counts[ order[ i ] ];
            ordered_node_types[ i ] = node_types[ order[ i ] ];
        }
        std::move( ordered_ia_arrays.begin(), ordered_ia_arrays.end(), ia_arrays.begin() + begin_pos );
        std::move( ordered_node_counts.begin(), ordered_node_counts.end(), node_counts.begin() + begin_pos );
        std::move( ordered_node_types.begin(), ordered_node_types.end(), node_types.begin() + begin_pos );
    }
}

TEST_CASE( "FPHeaderTable correctly functions", "[FPHeaderTable]" ) {
    cudaDeviceReset();

    const Item a = 0, b = 1, c = 2, d = 3, e = 4, f = 5, g = 6, h = 7, i = 8, j = 9, k = 10, l = 11, m = 12, n = 13,
               o = 14, p = 15, q = 16, r = 17, s = 18, t = 19, u = 20, v = 21, w = 22, x = 23, y = 24, z = 25;

    // each line represents a transaction
    Items trans {
        f, a, c, d, g, i, m, p,
        a, b, c, f, l, m, o,
        b, f, h, j, o,
        b, c, k, s, p,
        a, f, c, e, l, p, m, n
    };

    // start index of each transaction
    Indices indices { 0, 8, 15, 20, 25 };

    // number of items in each transaction
    Sizes sizes { 8, 7, 5, 5, 8 };

    // construct FPTransactionMap object
    size_type min_support = 3;
    FPTransMap trans_map( trans.cbegin(), indices.cbegin(), sizes.cbegin(), indices.size(), min_support );
    FPRadixTree radix_tree( trans_map );
    FPHeaderTable header_table( trans_map, radix_tree, min_support );
    cudaDeviceSynchronize();

    SECTION( "FPHeaderTable correctly initialize a header table from binary radix tree" ) {
        /* expected outcome:
                      I0:root (0,3)/1
                      ||
                      ++----------------+
                      |                 |
                      v                 |
            I1:010 (0,1)/0              I2:1 (2,3)/2
            ||                          ||
            ++------------+             ++------------+
            |             |             |             |
            v             v             v             v
            L0 x1         L1 x1         L2 x2         L3 x1
            ___           ___           _             _
            01000100      01011000      10111100      11101100

           Node  Item:Count  IA     NodeCount  Inner/Leaf
           H0:   a:3         2      3          I
           H1:   b:3         1,3    2,1        I,L
           H2:   m:3         2,3    2,1        L,L
           H3:   p:3         1,2    1,2        L,L
           H4:   c:4         1,2,3  1,2,1      L,L,L
           H5:   f:4         0,2,3  1,2,1      L,L,L
         */

        REQUIRE( header_table.size() == 6 );
        REQUIRE( header_table.ia_size() == 4 );

        Items items = header_table.get_items();
        REQUIRE( items.size() == 6 );
        CHECK( items[ 0 ] == a );
        CHECK( items[ 1 ] == b );
        CHECK( items[ 2 ] == m );
        CHECK( items[ 3 ] == p );
        CHECK( items[ 4 ] == c );
        CHECK( items[ 5 ] == f );

        Sizes items_freqs = header_table.get_counts();
        REQUIRE( items_freqs.size() == 6 );
        CHECK( items_freqs[ 0 ] == 3 );
        CHECK( items_freqs[ 1 ] == 3 );
        CHECK( items_freqs[ 2 ] == 3 );
        CHECK( items_freqs[ 3 ] == 3 );
        CHECK( items_freqs[ 4 ] == 4 );
        CHECK( items_freqs[ 5 ] == 4 );

        Sizes ia_sizes = header_table.get_ia_sizes();
        REQUIRE( ia_sizes.size() == 6 );
        CHECK( ia_sizes[ 0 ] == 1 );
        CHECK( ia_sizes[ 1 ] == 2 );
        CHECK( ia_sizes[ 2 ] == 2 );
        CHECK( ia_sizes[ 3 ] == 2 );
        CHECK( ia_sizes[ 4 ] == 3 );
        CHECK( ia_sizes[ 5 ] == 3 );

        Indices ia_array;
        Sizes node_counts;
        NodeTypes node_type;

        ia_array = header_table.get_ia_array( 0 );
        REQUIRE( ia_array.size() == 1 );
        node_counts = header_table.get_node_counts( 0 );
        REQUIRE( node_counts.size() == 1 );
        node_type = header_table.get_node_types( 0 );
        REQUIRE( node_type.size() == 1 );
        CHECK( ia_array[ 0 ] == 2 );
        CHECK( node_counts[ 0 ] == 3 );
        CHECK( node_type[ 0 ] == NodeType::InnerNode );

        ia_array = header_table.get_ia_array( 1 );
        REQUIRE( ia_array.size() == 2 );
        node_counts = header_table.get_node_counts( 1 );
        REQUIRE( node_counts.size() == 2 );
        node_type = header_table.get_node_types( 1 );
        REQUIRE( node_type.size() == 2 );
        sort_results( { 4 }, ia_array, node_counts, node_type );
        CHECK( ia_array[ 0 ] == 1 );
        CHECK( ia_array[ 1 ] == 3 );
        CHECK( node_counts[ 0 ] == 2 );
        CHECK( node_counts[ 1 ] == 1 );
        CHECK( node_type[ 0 ] == NodeType::InnerNode );
        CHECK( node_type[ 1 ] == NodeType::LeafNode );

        ia_array = header_table.get_ia_array( 2 );
        REQUIRE( ia_array.size() == 2 );
        node_counts = header_table.get_node_counts( 2 );
        REQUIRE( node_counts.size() == 2 );
        node_type = header_table.get_node_types( 2 );
        REQUIRE( node_type.size() == 2 );
        sort_results( { 4 }, ia_array, node_counts, node_type );
        CHECK( ia_array[ 0 ] == 2 );
        CHECK( ia_array[ 1 ] == 3 );
        CHECK( node_counts[ 0 ] == 2 );
        CHECK( node_counts[ 1 ] == 1 );
        CHECK( node_type[ 0 ] == NodeType::LeafNode );
        CHECK( node_type[ 1 ] == NodeType::LeafNode );

        ia_array = header_table.get_ia_array( 3 );
        REQUIRE( ia_array.size() == 2 );
        node_counts = header_table.get_node_counts( 3 );
        REQUIRE( node_counts.size() == 2 );
        node_type = header_table.get_node_types( 3 );
        REQUIRE( node_type.size() == 2 );
        sort_results( { 4 }, ia_array, node_counts, node_type );
        CHECK( ia_array[ 0 ] == 1 );
        CHECK( ia_array[ 1 ] == 2 );
        CHECK( node_counts[ 0 ] == 1 );
        CHECK( node_counts[ 1 ] == 2 );
        CHECK( node_type[ 0 ] == NodeType::LeafNode );
        CHECK( node_type[ 1 ] == NodeType::LeafNode );

        ia_array = header_table.get_ia_array( 4 );
        REQUIRE( ia_array.size() == 3 );
        node_counts = header_table.get_node_counts( 4 );
        REQUIRE( node_counts.size() == 3 );
        node_type = header_table.get_node_types( 4 );
        REQUIRE( node_type.size() == 3 );
        sort_results( { 4 }, ia_array, node_counts, node_type );
        CHECK( ia_array[ 0 ] == 1 );
        CHECK( ia_array[ 1 ] == 2 );
        CHECK( ia_array[ 2 ] == 3 );
        CHECK( node_counts[ 0 ] == 1 );
        CHECK( node_counts[ 1 ] == 2 );
        CHECK( node_counts[ 2 ] == 1 );
        CHECK( node_type[ 0 ] == NodeType::LeafNode );
        CHECK( node_type[ 1 ] == NodeType::LeafNode );
        CHECK( node_type[ 2 ] == NodeType::LeafNode );

        ia_array = header_table.get_ia_array( 5 );
        REQUIRE( ia_array.size() == 3 );
        node_counts = header_table.get_node_counts( 5 );
        REQUIRE( node_counts.size() == 3 );
        node_type = header_table.get_node_types( 5 );
        REQUIRE( node_type.size() == 3 );
        sort_results( { 4 }, ia_array, node_counts, node_type );
        CHECK( ia_array[ 0 ] == 0 );
        CHECK( ia_array[ 1 ] == 2 );
        CHECK( ia_array[ 2 ] == 3 );
        CHECK( node_counts[ 0 ] == 1 );
        CHECK( node_counts[ 1 ] == 2 );
        CHECK( node_counts[ 2 ] == 1 );
        CHECK( node_type[ 0 ] == NodeType::LeafNode );
        CHECK( node_type[ 1 ] == NodeType::LeafNode );
        CHECK( node_type[ 2 ] == NodeType::LeafNode );
    }

    SECTION( "FPHeaderTable correctly constructs sub header tables" ) {
        /* expected outcome:
                      I0:root (0,3)/1
                      ||
                      ++----------------+
                      |                 |
                      v                 |
            I1:010 (0,1)/0              I2:1 (2,3)/2
            ||                          ||
            ++------------+             ++------------+
            |             |             |             |
            v             v             v             v
            L0 x1         L1 x1         L2 x2         L3 x1
            ___           ___           _             _
            01000100      01011000      10111100      11101100

           Node  Item:Count  IA     NodeCount  Inner/Leaf
           H0:   a:3         2      3          I
           H1:   b:3         1,3    2,1        I,L
           H2:   m:3         2,3    2,1        L,L
           H3:   p:3         1,2    1,2        L,L
           H4:   c:4         1,2,3  1,2,1      L,L,L
           H5:   f:4         0,2,3  1,2,1      L,L,L

           Sub Table for H0
           Node  Item:Count  IA     NodeCount  Inner/Leaf

           Sub Table for H1
           Node  Item:Count  IA     NodeCount  Inner/Leaf
           H0:   a:1         2      1          I

           Sub Table for H2
           Node  Item:Count  IA     NodeCount  Inner/Leaf
           H0:   a:3         2      3          I
           H1:   b:1         3      1          L

           Sub Table for H3
           Node  Item:Count  IA     NodeCount  Inner/Leaf
           H0:   a:2         2      2          I
           H1:   b:1         1      1          I
           H2:   m:2         2      2          L

           Sub Table for H4
           Node  Item:Count  IA     NodeCount  Inner/Leaf
           H0:   a:3         2      3          I
           H1:   b:2         1,3    1,1        I,L
           H2:   m:3         2,3    2,1        L,L
           H3:   p:3         1,2    1,2        L,L

           Sub Table for H5
           Node  Item:Count  IA     NodeCount  Inner/Leaf
           H0:   a:3         2      3          I
           H1:   b:2         1,3    1,1        I,L
           H2:   m:3         2,3    2,1        L,L
           H3:   p:2         2      2          L
           H4:   c:3         2,3    2,1        L,L
         */

        Items items;
        Sizes counts, ia_sizes, node_counts;
        Indices ia_arrays;
        NodeTypes node_types;

        test_sub_header_table( trans_map, radix_tree, header_table, min_support, 0, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
        sort_results( ia_sizes, ia_arrays, node_counts, node_types );
        REQUIRE( items.size() == 6 );
        CHECK( items[ 0 ] == a );
        CHECK( items[ 1 ] == b );
        CHECK( items[ 2 ] == m );
        CHECK( items[ 3 ] == p );
        CHECK( items[ 4 ] == c );
        CHECK( items[ 5 ] == f );
        REQUIRE( counts.size() == 6 );
        CHECK( counts[ 0 ] == 0 );
        CHECK( counts[ 1 ] == 0 );
        CHECK( counts[ 2 ] == 0 );
        CHECK( counts[ 3 ] == 0 );
        CHECK( counts[ 4 ] == 0 );
        CHECK( counts[ 5 ] == 0 );
        REQUIRE( ia_sizes.size() == 6 );
        CHECK( ia_sizes[ 0 ] == 0 );
        CHECK( ia_sizes[ 1 ] == 0 );
        CHECK( ia_sizes[ 2 ] == 0 );
        CHECK( ia_sizes[ 3 ] == 0 );
        CHECK( ia_sizes[ 4 ] == 0 );
        CHECK( ia_sizes[ 5 ] == 0 );

        test_sub_header_table( trans_map, radix_tree, header_table, min_support, 1, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
        sort_results( ia_sizes, ia_arrays, node_counts, node_types );
        REQUIRE( items.size() == 6 );
        CHECK( items[ 0 ] == a );
        CHECK( items[ 1 ] == b );
        CHECK( items[ 2 ] == m );
        CHECK( items[ 3 ] == p );
        CHECK( items[ 4 ] == c );
        CHECK( items[ 5 ] == f );
        REQUIRE( counts.size() == 6 );
        CHECK( counts[ 0 ] == 1 );
        CHECK( counts[ 1 ] == 0 );
        CHECK( counts[ 2 ] == 0 );
        CHECK( counts[ 3 ] == 0 );
        CHECK( counts[ 4 ] == 0 );
        CHECK( counts[ 5 ] == 0 );
        REQUIRE( ia_sizes.size() == 6 );
        CHECK( ia_sizes[ 0 ] == 1 );
        CHECK( ia_sizes[ 1 ] == 0 );
        CHECK( ia_sizes[ 2 ] == 0 );
        CHECK( ia_sizes[ 3 ] == 0 );
        CHECK( ia_sizes[ 4 ] == 0 );
        CHECK( ia_sizes[ 5 ] == 0 );
        REQUIRE( ia_arrays.size() == 24 );
        CHECK( ia_arrays[ 0 ] == 2 );
        REQUIRE( node_counts.size() == 24 );
        CHECK( node_counts[ 0 ] == 1 );
        REQUIRE( node_types.size() == 24 );
        CHECK( node_types[ 0 ] == NodeType::InnerNode );

        test_sub_header_table( trans_map, radix_tree, header_table, min_support, 2, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
        sort_results( ia_sizes, ia_arrays, node_counts, node_types );
        REQUIRE( items.size() == 6 );
        CHECK( items[ 0 ] == a );
        CHECK( items[ 1 ] == b );
        CHECK( items[ 2 ] == m );
        CHECK( items[ 3 ] == p );
        CHECK( items[ 4 ] == c );
        CHECK( items[ 5 ] == f );
        REQUIRE( counts.size() == 6 );
        CHECK( counts[ 0 ] == 3 );
        CHECK( counts[ 1 ] == 1 );
        CHECK( counts[ 2 ] == 0 );
        CHECK( counts[ 3 ] == 0 );
        CHECK( counts[ 4 ] == 0 );
        CHECK( counts[ 5 ] == 0 );
        REQUIRE( ia_sizes.size() == 6 );
        CHECK( ia_sizes[ 0 ] == 1 );
        CHECK( ia_sizes[ 1 ] == 1 );
        CHECK( ia_sizes[ 2 ] == 0 );
        CHECK( ia_sizes[ 3 ] == 0 );
        CHECK( ia_sizes[ 4 ] == 0 );
        CHECK( ia_sizes[ 5 ] == 0 );
        REQUIRE( ia_arrays.size() == 24 );
        CHECK( ia_arrays[ 0 ] == 2 );
        CHECK( ia_arrays[ 1 ] == 0 );
        CHECK( ia_arrays[ 2 ] == 0 );
        CHECK( ia_arrays[ 3 ] == 0 );
        CHECK( ia_arrays[ 4 ] == 3 );
        CHECK( ia_arrays[ 5 ] == 0 );
        CHECK( ia_arrays[ 6 ] == 0 );
        CHECK( ia_arrays[ 7 ] == 0 );
        REQUIRE( node_counts.size() == 24 );
        CHECK( node_counts[ 0 ] == 3 );
        CHECK( node_counts[ 4 ] == 1 );
        REQUIRE( node_types.size() == 24 );
        CHECK( node_types[ 0 ] == NodeType::InnerNode );
        CHECK( node_types[ 4 ] == NodeType::LeafNode );

        test_sub_header_table( trans_map, radix_tree, header_table, min_support, 3, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
        sort_results( ia_sizes, ia_arrays, node_counts, node_types );
        REQUIRE( items.size() == 6 );
        CHECK( items[ 0 ] == a );
        CHECK( items[ 1 ] == b );
        CHECK( items[ 2 ] == m );
        CHECK( items[ 3 ] == p );
        CHECK( items[ 4 ] == c );
        CHECK( items[ 5 ] == f );
        REQUIRE( counts.size() == 6 );
        CHECK( counts[ 0 ] == 2 );
        CHECK( counts[ 1 ] == 1 );
        CHECK( counts[ 2 ] == 2 );
        CHECK( counts[ 3 ] == 0 );
        CHECK( counts[ 4 ] == 0 );
        CHECK( counts[ 5 ] == 0 );
        REQUIRE( ia_sizes.size() == 6 );
        CHECK( ia_sizes[ 0 ] == 1 );
        CHECK( ia_sizes[ 1 ] == 1 );
        CHECK( ia_sizes[ 2 ] == 1 );
        CHECK( ia_sizes[ 3 ] == 0 );
        CHECK( ia_sizes[ 4 ] == 0 );
        CHECK( ia_sizes[ 5 ] == 0 );
        REQUIRE( ia_arrays.size() == 24 );
        CHECK( ia_arrays[ 0 ] == 2 );
        CHECK( ia_arrays[ 4 ] == 1 );
        CHECK( ia_arrays[ 8 ] == 2 );
        REQUIRE( node_counts.size() == 24 );
        CHECK( node_counts[ 0 ] == 2 );
        CHECK( node_counts[ 4 ] == 1 );
        CHECK( node_counts[ 8 ] == 2 );
        REQUIRE( node_types.size() == 24 );
        CHECK( node_types[ 0 ] == NodeType::InnerNode );
        CHECK( node_types[ 4 ] == NodeType::InnerNode );
        CHECK( node_types[ 8 ] == NodeType::LeafNode );

        test_sub_header_table( trans_map, radix_tree, header_table, min_support, 4, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
        sort_results( ia_sizes, ia_arrays, node_counts, node_types );
        REQUIRE( items.size() == 6 );
        CHECK( items[ 0 ] == a );
        CHECK( items[ 1 ] == b );
        CHECK( items[ 2 ] == m );
        CHECK( items[ 3 ] == p );
        CHECK( items[ 4 ] == c );
        CHECK( items[ 5 ] == f );
        REQUIRE( counts.size() == 6 );
        CHECK( counts[ 0 ] == 3 );
        CHECK( counts[ 1 ] == 2 );
        CHECK( counts[ 2 ] == 3 );
        CHECK( counts[ 3 ] == 3 );
        CHECK( counts[ 4 ] == 0 );
        CHECK( counts[ 5 ] == 0 );
        REQUIRE( ia_sizes.size() == 6 );
        CHECK( ia_sizes[ 0 ] == 1 );
        CHECK( ia_sizes[ 1 ] == 2 );
        CHECK( ia_sizes[ 2 ] == 2 );
        CHECK( ia_sizes[ 3 ] == 2 );
        CHECK( ia_sizes[ 4 ] == 0 );
        CHECK( ia_sizes[ 5 ] == 0 );
        REQUIRE( ia_arrays.size() == 24 );
        CHECK( ia_arrays[ 0 ] == 2 );
        CHECK( ia_arrays[ 4 ] == 1 );
        CHECK( ia_arrays[ 5 ] == 3 );
        CHECK( ia_arrays[ 8 ] == 2 );
        CHECK( ia_arrays[ 9 ] == 3 );
        CHECK( ia_arrays[ 12 ] == 1 );
        CHECK( ia_arrays[ 13 ] == 2 );
        REQUIRE( node_counts.size() == 24 );
        CHECK( node_counts[ 0 ] == 3 );
        CHECK( node_counts[ 4 ] == 1 );
        CHECK( node_counts[ 5 ] == 1 );
        CHECK( node_counts[ 8 ] == 2 );
        CHECK( node_counts[ 9 ] == 1 );
        CHECK( node_counts[ 12 ] == 1 );
        CHECK( node_counts[ 13 ] == 2 );
        REQUIRE( node_types.size() == 24 );
        CHECK( node_types[ 0 ] == NodeType::InnerNode );
        CHECK( node_types[ 4 ] == NodeType::InnerNode );
        CHECK( node_types[ 5 ] == NodeType::LeafNode );
        CHECK( node_types[ 8 ] == NodeType::LeafNode );
        CHECK( node_types[ 9 ] == NodeType::LeafNode );
        CHECK( node_types[ 12 ] == NodeType::LeafNode );
        CHECK( node_types[ 13 ] == NodeType::LeafNode );

        test_sub_header_table( trans_map, radix_tree, header_table, min_support, 5, items, counts, ia_sizes, ia_arrays, node_counts, node_types );
        sort_results( ia_sizes, ia_arrays, node_counts, node_types );
        REQUIRE( items.size() == 6 );
        CHECK( items[ 0 ] == a );
        CHECK( items[ 1 ] == b );
        CHECK( items[ 2 ] == m );
        CHECK( items[ 3 ] == p );
        CHECK( items[ 4 ] == c );
        CHECK( items[ 5 ] == f );
        REQUIRE( counts.size() == 6 );
        CHECK( counts[ 0 ] == 3 );
        CHECK( counts[ 1 ] == 2 );
        CHECK( counts[ 2 ] == 3 );
        CHECK( counts[ 3 ] == 2 );
        CHECK( counts[ 4 ] == 3 );
        CHECK( counts[ 5 ] == 0 );
        REQUIRE( ia_sizes.size() == 6 );
        CHECK( ia_sizes[ 0 ] == 1 );
        CHECK( ia_sizes[ 1 ] == 2 );
        CHECK( ia_sizes[ 2 ] == 2 );
        CHECK( ia_sizes[ 3 ] == 1 );
        CHECK( ia_sizes[ 4 ] == 2 );
        CHECK( ia_sizes[ 5 ] == 0 );
        REQUIRE( ia_arrays.size() == 24 );
        CHECK( ia_arrays[ 0 ] == 2 );
        CHECK( ia_arrays[ 4 ] == 1 );
        CHECK( ia_arrays[ 5 ] == 3 );
        CHECK( ia_arrays[ 8 ] == 2 );
        CHECK( ia_arrays[ 9 ] == 3 );
        CHECK( ia_arrays[ 12 ] == 2 );
        CHECK( ia_arrays[ 16 ] == 2 );
        CHECK( ia_arrays[ 17 ] == 3 );
        REQUIRE( node_counts.size() == 24 );
        CHECK( node_counts[ 0 ] == 3 );
        CHECK( node_counts[ 4 ] == 1 );
        CHECK( node_counts[ 5 ] == 1 );
        CHECK( node_counts[ 8 ] == 2 );
        CHECK( node_counts[ 9 ] == 1 );
        CHECK( node_counts[ 12 ] == 2 );
        CHECK( node_counts[ 16 ] == 2 );
        CHECK( node_counts[ 17 ] == 1 );
        REQUIRE( node_types.size() == 24 );
        CHECK( node_types[ 0 ] == NodeType::InnerNode );
        CHECK( node_types[ 4 ] == NodeType::InnerNode );
        CHECK( node_types[ 5 ] == NodeType::LeafNode );
        CHECK( node_types[ 8 ] == NodeType::LeafNode );
        CHECK( node_types[ 9 ] == NodeType::LeafNode );
        CHECK( node_types[ 12 ] == NodeType::LeafNode );
        CHECK( node_types[ 16 ] == NodeType::LeafNode );
        CHECK( node_types[ 17 ] == NodeType::LeafNode );
    }
}

}
