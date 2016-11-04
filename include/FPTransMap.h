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

#ifndef CUDA_FP_GROWTH_FPTRANSMAP_H
#define CUDA_FP_GROWTH_FPTRANSMAP_H

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

namespace cuda_fp_growth {

#define cudaCheck( result ) { cudaAssert( (result), __FILE__, __LINE__ ); }
#define cudaCheckWithMessage( result, msg ) { cudaAssert( (result), __FILE__, __LINE__, msg ); }

inline void cudaAssert( cudaError_t code, const std::string& file, int line, const std::string& msg = "" )
{
    if ( code != cudaSuccess ) {
        std::cerr << ( msg.length() == 0 ? std::string( cudaGetErrorString( code ) ) : msg ) << ": " << file << " " << line << std::endl;
#ifndef NDEBUG
        exit( code );
#endif // NDEBUG
    }
}

// The default data type for use with CUDA
using cuda_int = int;
using cuda_uint = unsigned int;
using cuda_real = float;

using Item = cuda_uint;
using Items = std::vector<Item>;
using DItems = thrust::device_vector<Item>;

static const Item NullItem = std::numeric_limits<Item>::max();

using size_type = cuda_uint;
using Sizes = std::vector<size_type>;
using DSizes = thrust::device_vector<size_type>;

using index_type = cuda_uint;
using Indices = std::vector<index_type>;
using DIndices = thrust::device_vector<index_type>;

using BitBlock = uint8_t;
using BitBlocks = std::vector<BitBlock>;
using DBitBlocks = thrust::device_vector<BitBlock>;

class FPTransMap
{
    public:
        /** \brief Constructs a transaction bitmap from transaction data set
         *
         * The transaction data are stored in a vector in row-major order (i.e. items in same transaction are stored together and followed by items of next transaction).
         * Another two vectors are used to specify the begin index of each transaction. For example, if there are 3 transactions:
         *
         * 1. f, a, c, d, g, i, m, p,
         * 2. a, b, c, f, l, m, o,
         * 3. b, f, h, j, o,
         * 4. b, c, k, s, p,
         * 5. a, f, c, e, l, p, m, n
         *
         * The items vector will be ( f, a, c, d, g, i, m, p, a, b, c, f, l, m, o, b, f, h, j, o, b, c, k, s, p, a, f, c, e, l, p, m, n ),
         * the indices vector will be ( 0, 8, 15, 20, 25 ),
         * the sizes vector will be ( 8, 7, 5, 5, 8 ) and the n_trans will be 5.
         *
         * The transactions are scanned once and the frequent items are extracted and then sorted by frequency in descending order. The above transactions have 6 frequent items:
         * ( a, b, m, p, c, f ). A bitmap is generated to encode the relations between the transactions and frequent items:
         *
         * ```
         *    T1 T2 T3 T4 T5
         * a   1  1  0  0  1  0  0  0
         * b   0  1  1  1  0  0  0  0
         * m   1  1  0  0  1  0  0  0
         * p   1  0  0  1  1  0  0  0
         * c   1  1  0  1  1  0  0  0
         * f   1  1  1  0  1  0  0  0
         *
         * ```
         *
         * Note that if the number of transactions is not a multiple of 8, the remaining bits are padded with 0. The bitmap is then transposed:
         *
         * ```
         *       a    b    m    p    c    f
         * T1    1    0    1    1    1    1    0    0
         * T2    1    1    1    0    1    1    0    0
         * T3    0    1    0    0    0    1    0    0
         * T4    0    1    0    1    1    0    0    0
         * T5    1    0    1    1    1    1    0    0
         *       0    0    0    0    0    0    0    0
         *       0    0    0    0    0    0    0    0
         *       0    0    0    0    0    0    0    0
         *
         * ```
         *
         * Finally, the transposed bitmap is checked for duplicate transactions. All duplicates are merged so the bitmap contains unique keys.
         * The padding rows (e.g. the last 3 rows in above example) are also removed.
         * The counts of each unique transaction are stored and the `transaction_counts()` function returns a vector of counts for each unique key.
         *
         * \param trans_begin iterator pointing to the start of the items vector
         * \param indices_begin iterator pointing to the start of the indices vector
         * \param sizes_begin iterator pointing to the start of the sizes vector
         * \param n_trans total number of transactions
         * \param min_support the minimum support required
         *
         */
        FPTransMap( Items::const_iterator trans_begin, Indices::const_iterator indices_begin, Sizes::const_iterator sizes_begin, size_type n_trans, size_type min_support );

        /** \brief Constructs a transaction bitmap from transaction data set for association rules mining
         *
         * \copydetails FPTransMap( Items::const_iterator, Indices::const_iterator, Sizes::const_iterator, size_type, size_type )
         *
         * For association rules mining, this constructor accepts an extra parameter:
         *     - rhs: the right hand side item. This item will be placed at the begining of the transaction bits regardless its frequency so it will be mined after all other items.
         *
         * The definition of `min_support` is slight different from the one used for frequent pattern mining.
         * For association rule mining the support for rule `A->B` is defined as `count(A)` rather than `count(AB)` as the case in frequent pattern mining.
         *
         * \param trans_begin iterator pointing to the start of the items vector
         * \param indices_begin iterator pointing to the start of the indices vector
         * \param sizes_begin iterator pointing to the start of the sizes vector
         * \param n_trans total number of transactions
         * \param rhs the item on the right hand side of an association rule. Currently only a single right hand side item is supported.
         * \param min_support the minimum support required
         *
         */
        FPTransMap( Items::const_iterator trans_begin, Indices::const_iterator indices_begin, Sizes::const_iterator sizes_begin, size_type n_trans, const Item& rhs, size_type min_support );

        virtual ~FPTransMap() = default;

        inline size_type size() const { return _trans_counts.size(); }

        inline const DItems& frequent_items() const { return _freq_items; }

        inline const DSizes& items_frequency() const { return _freq_items_counts; }

        inline size_type max_frequency() const { return _max_frequency; }

        inline const DBitBlocks& bitmap() const { return _trans_map; }

        inline const DSizes& transaction_counts() const { return _trans_counts; }

        inline size_type blocks_per_transaction() const { return _blocks_per_trans; }

        inline const Item& rhs() const { return _rhs; }

    private:
        const Item _rhs;
        DItems _freq_items;
        DSizes _freq_items_counts;
        DBitBlocks _trans_map;
        DSizes _trans_counts;
        size_type _blocks_per_trans, _max_frequency;

        void build_transaction_map( Items::const_iterator trans_begin, Indices::const_iterator indices_begin, Sizes::const_iterator sizes_begin, size_type n_trans, const Item& rhs, size_type min_support, Items& freq_items, Sizes& freq_items_counts, BitBlocks& bitmap );

        void transpose_bitmap( BitBlocks& bitmap, size_type blocks_per_row );

        void sort_and_merge( BitBlocks& bitmap, size_type blocks_per_trans, Sizes& counts );
};

}

#endif // CUDA_FP_GROWTH_FPTRANSMAP_H
