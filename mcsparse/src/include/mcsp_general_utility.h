#ifndef MCSP_GENERAL_UTILITY_H
#define MCSP_GENERAL_UTILITY_H

#define ALIGN(n, size) (((n) + (size)-1) / (size) * (size))

#define CEIL(n, size) (((n) + (size)-1) / (size))

#define THRESHOLD(value, eps) ((value) != static_cast<valType>(0) ? (value) : eps)

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir, block_dim) \
    ((dir == MCSPARSE_DIRECTION_ROW) ? BSR_IND_R(j, bi, bj, block_dim) : BSR_IND_C(j, bi, bj, block_dim))
#define BSR_IND_R(j, bi, bj, block_dim) (block_dim * block_dim * (j) + (bi)*block_dim + (bj))
#define BSR_IND_C(j, bi, bj, block_dim) (block_dim * block_dim * (j) + (bi) + (bj)*block_dim)

#endif  // MCSP_GENERAL_UTILITY_H