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

#include "FPRadixTree.h"

namespace cuda_fp_growth {

TEST_CASE( "FPRadixTree correctly functions", "[FPRadixTree]" ) {
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
    FPTransMap fp_trans_map( trans.cbegin(), indices.cbegin(), sizes.cbegin(), indices.size(), 3 );
    FPRadixTree fp_radix_tree( fp_trans_map );

    SECTION( "FPRadixTree correctly constructs a binary radix tree" ) {
        const DInnerNodes& d_inner_nodes = fp_radix_tree.inner_nodes();
        const DLeafNodes& d_leaf_nodes = fp_radix_tree.leaf_nodes();
        InnerNodes inner_nodes( d_inner_nodes.cbegin(), d_inner_nodes.cend() );
        LeafNodes leaf_nodes( d_leaf_nodes.cbegin(), d_leaf_nodes.cend() );

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
         */

        REQUIRE( inner_nodes.size() == 3 );
        CHECK( inner_nodes[ 0 ].parent_idx == 0 );
        CHECK( inner_nodes[ 0 ].range_start == 0 );
        CHECK( inner_nodes[ 0 ].range_end == 3 );
        CHECK( inner_nodes[ 0 ].left_is_leaf == false );
        CHECK( inner_nodes[ 0 ].right_is_leaf == false );
        CHECK( inner_nodes[ 0 ].left_idx == 1 );
        CHECK( inner_nodes[ 0 ].right_idx == 2 );
        CHECK( inner_nodes[ 0 ].prefix_length == 0 );
        CHECK( inner_nodes[ 0 ].trans_count == 5 );

        CHECK( inner_nodes[ 1 ].parent_idx == 0 );
        CHECK( inner_nodes[ 1 ].range_start == 0 );
        CHECK( inner_nodes[ 1 ].range_end == 1 );
        CHECK( inner_nodes[ 1 ].left_is_leaf == true );
        CHECK( inner_nodes[ 1 ].right_is_leaf == true );
        CHECK( inner_nodes[ 1 ].left_idx == 0 );
        CHECK( inner_nodes[ 1 ].right_idx == 1 );
        CHECK( inner_nodes[ 1 ].prefix_length == 3 );
        CHECK( inner_nodes[ 1 ].trans_count == 2 );

        CHECK( inner_nodes[ 2 ].parent_idx == 0 );
        CHECK( inner_nodes[ 2 ].range_start == 2 );
        CHECK( inner_nodes[ 2 ].range_end == 3 );
        CHECK( inner_nodes[ 2 ].left_is_leaf == true );
        CHECK( inner_nodes[ 2 ].right_is_leaf == true );
        CHECK( inner_nodes[ 2 ].left_idx == 2 );
        CHECK( inner_nodes[ 2 ].right_idx == 3 );
        CHECK( inner_nodes[ 2 ].prefix_length == 1 );
        CHECK( inner_nodes[ 2 ].trans_count == 3 );

        REQUIRE( leaf_nodes.size() == 4 );
        CHECK( leaf_nodes[ 0 ].parent_idx == 1 );
        CHECK( leaf_nodes[ 0 ].trans_count == 1 );

        CHECK( leaf_nodes[ 1 ].parent_idx == 1 );
        CHECK( leaf_nodes[ 1 ].trans_count == 1 );

        CHECK( leaf_nodes[ 2 ].parent_idx == 2 );
        CHECK( leaf_nodes[ 2 ].trans_count == 2 );

        CHECK( leaf_nodes[ 3 ].parent_idx == 2 );
        CHECK( leaf_nodes[ 3 ].trans_count == 1 );
    }
}

}
