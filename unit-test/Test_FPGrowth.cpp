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

#include "FPGrowth.h"

namespace cuda_fp_growth {

size_type pattern_count( const std::vector<cuda_uint>& buffer )
{
    index_type i = 0;
    size_type pattern_count = 0;
    while ( i < buffer.size() ) {
        ++pattern_count;
        i += ( buffer[i] / sizeof( cuda_uint ) );
    }
    return pattern_count;
}

bool pattern_exists( const std::vector<cuda_uint>& buffer, const std::vector<Item>& pattern, const size_type support, const cuda_real confidence = 0.0f )
{
    index_type i = 0;
    while ( i < buffer.size() ) {
        size_type length = buffer[ i ] / sizeof( cuda_uint );
        size_type offset = ( confidence > 0.0f ? 3 : 2 );
        bool exists = true;
        exists &= ( pattern.size() == length - offset );
        exists &= ( buffer[ i + 1 ] == support );
        exists &= ( std::equal( pattern.begin(), pattern.end(), buffer.begin() + i + offset ) );
        if ( confidence > 0.0f ) {
            const cuda_uint* ptr = &buffer[ i + 2 ];
            exists &= ( std::abs( *( reinterpret_cast<const cuda_real*>( ptr ) ) - confidence ) < 0.0001 );
        }
        if ( exists ) return true;

        i += ( buffer[i] / sizeof( cuda_uint ) );
    }
    return false;
}

TEST_CASE( "FPGrowth correctly functions", "[FPGrowth]" ) {
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

    SECTION( "FPGrowth correctly finds all frequent patterns") {
        FPTransMap trans_map( trans.cbegin(), indices.cbegin(), sizes.cbegin(), indices.size(), min_support );
        FPRadixTree radix_tree( trans_map );
        FPHeaderTable ht( trans_map, radix_tree, min_support );
        FPGrowth fp( trans_map, radix_tree, ht, min_support );

        std::vector<cuda_uint> buffer( 1024 );
        size_type buffer_size = sizeof( cuda_uint ) * buffer.size();
        fp.mine_frequent_patterns( &buffer[0], buffer_size );
        buffer.resize( buffer_size / sizeof( cuda_uint ) );

        REQUIRE( pattern_count( buffer ) == 18 );
        CHECK( pattern_exists( buffer, { a }, 3 ) );
        CHECK( pattern_exists( buffer, { b }, 3 ) );
        CHECK( pattern_exists( buffer, { m }, 3 ) );
        CHECK( pattern_exists( buffer, { p }, 3 ) );
        CHECK( pattern_exists( buffer, { c }, 4 ) );
        CHECK( pattern_exists( buffer, { f }, 4 ) );
        CHECK( pattern_exists( buffer, { c, p }, 3 ) );
        CHECK( pattern_exists( buffer, { m, a }, 3 ) );
        CHECK( pattern_exists( buffer, { c, m }, 3 ) );
        CHECK( pattern_exists( buffer, { f, m }, 3 ) );
        CHECK( pattern_exists( buffer, { f, a }, 3 ) );
        CHECK( pattern_exists( buffer, { c, a }, 3 ) );
        CHECK( pattern_exists( buffer, { f, c }, 3 ) );
        CHECK( pattern_exists( buffer, { c, m, a }, 3 ) );
        CHECK( pattern_exists( buffer, { f, m, a }, 3 ) );
        CHECK( pattern_exists( buffer, { f, c, m }, 3 ) );
        CHECK( pattern_exists( buffer, { f, c, a }, 3 ) );
        CHECK( pattern_exists( buffer, { f, c, m, a }, 3 ) );
    }

    SECTION( "FPGrowth correctly finds all association rules") {
        Item rhs = m;
        FPTransMap trans_map( trans.cbegin(), indices.cbegin(), sizes.cbegin(), indices.size(), rhs, min_support );
        FPRadixTree radix_tree( trans_map );
        FPHeaderTable ht( trans_map, radix_tree, min_support );
        FPGrowth fp( trans_map, radix_tree, ht, min_support );

        std::vector<cuda_uint> buffer( 1024 );
        size_type buffer_size = sizeof( cuda_uint ) * buffer.size();
        fp.mine_association_rules( 1.0, &buffer[0], buffer_size );
        buffer.resize( buffer_size / sizeof( cuda_uint ) );

        REQUIRE( pattern_count( buffer ) == 5 );
        CHECK( pattern_exists( buffer, { a }, 3, 1.0 ) );
        CHECK( pattern_exists( buffer, { c, a }, 3, 1.0 ) );
        CHECK( pattern_exists( buffer, { f, a }, 3, 1.0 ) );
        CHECK( pattern_exists( buffer, { f, c }, 3, 1.0 ) );
        CHECK( pattern_exists( buffer, { f, c, a }, 3, 1.0 ) );
    }
}

}
