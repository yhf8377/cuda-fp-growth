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

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>

#include "FPTransMap.h"

namespace cuda_fp_growth {

FPTransMap::FPTransMap( Items::const_iterator trans_begin, Indices::const_iterator indices_begin, Sizes::const_iterator sizes_begin, size_type n_trans, size_type min_support )
{
    // determine number of blocks required for the given transaction count
    assert( std::numeric_limits<BitBlock>::digits == 8 );
    size_type block_size = std::numeric_limits<BitBlock>::digits;
    size_type blocks_per_row = ( n_trans / block_size ) + ( n_trans % block_size > 0 ? 1 : 0 );

    // convert transactions to items map
    Items freq_items;
    Sizes freq_items_counts;
    BitBlocks trans_map;
    build_transaction_map( trans_begin, indices_begin, sizes_begin, n_trans, min_support, freq_items, freq_items_counts, trans_map );
    _max_frequency = *( std::max_element( freq_items_counts.begin(), freq_items_counts.end() ) );

    // transpose items map into transactions map
    transpose_bitmap( trans_map, blocks_per_row );

    // sort transactions in lexicographical order and merge duplicates
    Sizes counts;
    _blocks_per_trans = size_type( freq_items.size() / 8 ) + ( freq_items.size() % 8 != 0 ? 1 : 0 );
    sort_and_merge( trans_map, _blocks_per_trans, counts );

    // load data onto GPU
    _freq_items = DItems( freq_items );
    _freq_items_counts = DSizes( freq_items_counts );
    _trans_map = DBitBlocks( trans_map );
    _trans_counts = DSizes( counts );
}

void FPTransMap::build_transaction_map( Items::const_iterator trans_begin, Indices::const_iterator indices_begin, Sizes::const_iterator sizes_begin, size_type n_trans, size_type min_support, Items& freq_items, Sizes& freq_items_counts, BitBlocks& bitmap )
{
    // determine number of blocks required for the given transaction count
    assert( std::numeric_limits<BitBlock>::digits == 8 );
    size_type block_size = std::numeric_limits<BitBlock>::digits;
    size_type blocks_per_row = ( n_trans / block_size ) + ( n_trans % block_size > 0 ? 1 : 0 );

    // count items in transactions
    std::map<Item, BitBlocks> trans_map;
    std::map<Item, size_type> counts;
    for ( index_type i = 0; i < n_trans; ++i ) {
        Items::const_iterator first = trans_begin + indices_begin[ i ], last = first + sizes_begin[ i ];
        std::for_each( first, last, [&]( const Item& item ) {
            // look up current item in the map
            auto iter = trans_map.find( item );
            // if item does not exist, construct bit blocks to hold bits for all transactions
            if ( iter == trans_map.end() ) iter = trans_map.emplace( item, BitBlocks( blocks_per_row, 0 ) ).first;
            // set the bit for current transaction
            index_type block_idx = i / block_size;
            unsigned char bit_pos = block_size - i % block_size - 1;
            BitBlocks& blocks = iter->second;
            blocks[ block_idx ] |= 1 << bit_pos;
            // increase item count
            ++counts[ item ];
        } );
    }

    // copy frequent items and remove infrequent items
    freq_items.clear();
    for ( auto iter = counts.begin(), iter_end = counts.end(); iter != iter_end; ) {
        if ( iter->second >= min_support ) {
            freq_items.push_back( iter->first );
            ++iter;
        }
        else {
            iter = counts.erase( iter );
        }
    }

    // sort frequent items
    std::sort( freq_items.begin(), freq_items.end(), [&]( const Item& item_a, const Item& item_b ) {
        size_type count_a = counts.at( item_a ), count_b = counts.at( item_b );
        return ( count_a < count_b ) || ( count_a == count_b && item_a < item_b );
    } );

    // copy frequency count and transaction map in order
    freq_items_counts.clear();
    bitmap.clear();
    std::for_each( freq_items.cbegin(), freq_items.cend(), [&]( const Item& item ) {
        // copy frequency
        freq_items_counts.push_back( counts.at( item ) );
        // copy bitmap
        const BitBlocks& blocks = trans_map.at( item );
        bitmap.insert( bitmap.end(), blocks.begin(), blocks.end() );
    } );

    // pad items map if number of items is not multiple of 8
    size_type mod = freq_items.size() % 8;
    if ( mod != 0 ) {
        size_type pad_rows = 8 - mod;
        BitBlocks padding( pad_rows * blocks_per_row, 0 );
        bitmap.insert( bitmap.end(), std::make_move_iterator( padding.begin() ), std::make_move_iterator( padding.end() ) );
    }
}

void FPTransMap::transpose_bitmap( BitBlocks& bitmap, size_type blocks_per_row )
{
    assert( bitmap.size() % blocks_per_row == 0 );
    assert( ( bitmap.size() / blocks_per_row ) % 8 == 0 );

    BitBlocks transposed( bitmap.size() );
    size_type nrow = bitmap.size() / blocks_per_row, row_blocks = nrow / 8;
    for ( index_type i = 0; i < row_blocks; ++i ) {
        for ( index_type j = 0; j < blocks_per_row; ++j ) {
            uint64_t x = ( uint64_t( bitmap[   i * 8       * blocks_per_row + j ] ) << 56 ) |
                         ( uint64_t( bitmap[ ( i * 8 + 1 ) * blocks_per_row + j ] ) << 48 ) |
                         ( uint64_t( bitmap[ ( i * 8 + 2 ) * blocks_per_row + j ] ) << 40 ) |
                         ( uint64_t( bitmap[ ( i * 8 + 3 ) * blocks_per_row + j ] ) << 32 ) |
                         ( uint64_t( bitmap[ ( i * 8 + 4 ) * blocks_per_row + j ] ) << 24 ) |
                         ( uint64_t( bitmap[ ( i * 8 + 5 ) * blocks_per_row + j ] ) << 16 ) |
                         ( uint64_t( bitmap[ ( i * 8 + 6 ) * blocks_per_row + j ] ) <<  8 ) |
                         ( uint64_t( bitmap[ ( i * 8 + 7 ) * blocks_per_row + j ] ) );
            uint64_t y = (x & 0x8040201008040201LL) |
                        ((x & 0x0080402010080402LL) <<  7) |
                        ((x & 0x0000804020100804LL) << 14) |
                        ((x & 0x0000008040201008LL) << 21) |
                        ((x & 0x0000000080402010LL) << 28) |
                        ((x & 0x0000000000804020LL) << 35) |
                        ((x & 0x0000000000008040LL) << 42) |
                        ((x & 0x0000000000000080LL) << 49) |
                        ((x >>  7) & 0x0080402010080402LL) |
                        ((x >> 14) & 0x0000804020100804LL) |
                        ((x >> 21) & 0x0000008040201008LL) |
                        ((x >> 28) & 0x0000000080402010LL) |
                        ((x >> 35) & 0x0000000000804020LL) |
                        ((x >> 42) & 0x0000000000008040LL) |
                        ((x >> 49) & 0x0000000000000080LL);
            transposed[ ( j * 8 ) * row_blocks + i ]     = uint8_t( ( y >> 56 ) & 0xFF );
            transposed[ ( j * 8 + 1 ) * row_blocks + i ] = uint8_t( ( y >> 48 ) & 0xFF );
            transposed[ ( j * 8 + 2 ) * row_blocks + i ] = uint8_t( ( y >> 40 ) & 0xFF );
            transposed[ ( j * 8 + 3 ) * row_blocks + i ] = uint8_t( ( y >> 32 ) & 0xFF );
            transposed[ ( j * 8 + 4 ) * row_blocks + i ] = uint8_t( ( y >> 24 ) & 0xFF );
            transposed[ ( j * 8 + 5 ) * row_blocks + i ] = uint8_t( ( y >> 16 ) & 0xFF );
            transposed[ ( j * 8 + 6 ) * row_blocks + i ] = uint8_t( ( y >> 8 ) & 0xFF );
            transposed[ ( j * 8 + 7 ) * row_blocks + i ] = uint8_t( y & 0xFF );
        }
    }
    std::swap( bitmap, transposed );
}

void FPTransMap::sort_and_merge( BitBlocks& bitmap, size_type blocks_per_trans, Sizes& counts )
{
    assert( bitmap.size() % blocks_per_trans == 0 );
    size_type n_trans = bitmap.size() / blocks_per_trans;

    Indices indices( n_trans );
    std::iota( indices.begin(), indices.end(), 0 );

    // remove all zero rows (transactions that do not contain any frequent item)
    BitBlocks all_zero( blocks_per_trans, 0 );
    indices.erase( std::remove_if( indices.begin(), indices.end(), [&]( index_type idx ) {
        auto first = bitmap.cbegin() + idx * blocks_per_trans, last = first + blocks_per_trans;
        return !( std::lexicographical_compare( first, last, all_zero.cbegin(), all_zero.cend() ) || std::lexicographical_compare( all_zero.cbegin(), all_zero.cend(), first, last ) );
    } ), indices.end() );
    counts.resize( indices.size() );

    // sort transaction blocks
    std::sort( indices.begin(), indices.end(), [&]( index_type idx_a, index_type idx_b ) {
        auto first1 = bitmap.cbegin() + idx_a * blocks_per_trans, last1 = first1 + blocks_per_trans;
        auto first2 = bitmap.cbegin() + idx_b * blocks_per_trans, last2 = first2 + blocks_per_trans;
        return std::lexicographical_compare( first1, last1, first2, last2 );
    } );

    // scan and count duplicates
    index_type counts_idx = 0;
    for ( auto iter1 = indices.begin(), iter2 = iter1 + 1; iter2 != indices.end(); ) {
        ++counts[ counts_idx ];

        index_type offset1 = *iter1 * blocks_per_trans, offset2 = *iter2 * blocks_per_trans;
        auto first1 = bitmap.begin() + offset1, last1 = first1 + blocks_per_trans, first2 = bitmap.begin() + offset2, last2 = first2 + blocks_per_trans;
        // because we already sorted the transactions (i.e. transaction1 <= transaction2), we only check if transaction1 also >= transaction2 (i.e. if yes, they are equal).
        if ( !std::lexicographical_compare( first1, last1, first2, last2 ) ) {
            iter2 = indices.erase( iter2 );
        }
        else {
            iter1 = iter2++;
            ++counts_idx;
        }
    }
    ++counts[ counts_idx ];

    // copy remaining blocks
    BitBlocks new_bitmap( indices.size() * blocks_per_trans );
    for ( index_type idx = 0; idx < indices.size(); ++idx ) {
        auto source_first = bitmap.begin() + indices[ idx ] * blocks_per_trans, source_last = source_first + blocks_per_trans;
        auto dest_first = new_bitmap.begin() + idx * blocks_per_trans;
        std::move( source_first, source_last, dest_first );
    }
    counts.resize( indices.size() );
    std::swap( bitmap, new_bitmap );
}

}
