
#ifndef COMMON_MCSP_INTERNAL_HALF_HPP_
#define COMMON_MCSP_INTERNAL_HALF_HPP_

/**
 * \brief half datatype
 *
 * \details This structure implements the datatype for storing
 * half-precision floating-point numbers. The structure implements
 * assignment operators and type conversions.
 * 16 bits are being used in total: 1 sign bit, 5 bits for the exponent,
 * and the significand is being stored in 10 bits.
 * The total precision is 11 bits. There are 15361 representable
 * numbers within the interval [0.0, 1.0], endpoints included.
 * On average we have log10(2**11) ~ 3.311 decimal digits.
 *
 * \internal
 * \req IEEE 754-2008 compliant implementation of half-precision
 * floating-point numbers.
 * \endinternal
 */
struct __half;

/**
 * \brief half2 datatype
 *
 * \details This structure implements the datatype for storing two
 * half-precision floating-point numbers.
 * The structure implements assignment operators and type conversions.
 *
 * \internal
 * \req Vectorified version of half.
 * \endinternal
 */
struct __half2;

#endif  // COMMON_MCSP_INTERNAL_HALF_HPP_
