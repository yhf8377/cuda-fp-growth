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

#include "FPTransMap.h"

namespace cuda_fp_growth {

TEST_CASE( "FPTransMap correctly functions", "[FPTransMap]" ) {
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

    SECTION( "FPTransMap correctly identifies Frequent Items" ) {
        const DItems& d_freq_items = fp_trans_map.frequent_items();
        const DSizes& d_freq_items_counts = fp_trans_map.items_frequency();
        const Items freq_items( d_freq_items.begin(), d_freq_items.end() );
        const Sizes freq_items_counts( d_freq_items_counts.begin(), d_freq_items_counts.end() );

        // expected outcome: freq_items = [ a, b, m, p, c, f ]
        // expected outcome: freq_items_counts = [ 3, 3, 3, 3, 4, 4 ]

        REQUIRE( fp_trans_map.max_frequency() == 4 );

        REQUIRE( freq_items.size() == 6 );
        CHECK( freq_items[ 0 ] == a );
        CHECK( freq_items[ 1 ] == b );
        CHECK( freq_items[ 2 ] == m );
        CHECK( freq_items[ 3 ] == p );
        CHECK( freq_items[ 4 ] == c );
        CHECK( freq_items[ 5 ] == f );

        REQUIRE( freq_items_counts.size() == 6 );
        CHECK( freq_items_counts[ 0 ] == 3 );
        CHECK( freq_items_counts[ 1 ] == 3 );
        CHECK( freq_items_counts[ 2 ] == 3 );
        CHECK( freq_items_counts[ 3 ] == 3 );
        CHECK( freq_items_counts[ 4 ] == 4 );
        CHECK( freq_items_counts[ 5 ] == 4 );
    }

    SECTION( "FPTransMap correctly builds transaction bitmap" ) {
        const DBitBlocks& d_trans_map = fp_trans_map.bitmap();
        const DSizes& d_counts = fp_trans_map.transaction_counts();
        const BitBlocks trans_map( d_trans_map.begin(), d_trans_map.end() );
        const Sizes counts( d_counts.begin(), d_counts.end() );

        /* expected outcome:
                     a b m p c f
           Key0: 1   0 1 0 0 0 1 0 0    = 68
           Key1: 1   0 1 0 1 1 0 0 0    = 88
           Key2: 2   1 0 1 1 1 1 0 0    = 188
           Key3: 1   1 1 1 0 1 1 0 0    = 236
         */


        REQUIRE( trans_map.size() == 4 );
        CHECK( trans_map[ 0 ] == 68 );
        CHECK( trans_map[ 1 ] == 88 );
        CHECK( trans_map[ 2 ] == 188 );
        CHECK( trans_map[ 3 ] == 236 );

        REQUIRE( counts.size() == 4 );
        CHECK( counts[ 0 ] == 1 );
        CHECK( counts[ 1 ] == 1 );
        CHECK( counts[ 2 ] == 2 );
        CHECK( counts[ 3 ] == 1 );
    }
}

}
