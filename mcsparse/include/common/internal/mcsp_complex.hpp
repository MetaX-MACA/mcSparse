
#ifndef COMMON_MCSP_COMPLEX_H_
#define COMMON_MCSP_COMPLEX_H_

#if __cplusplus < 201402L
// Only include minimal definitions of complex data type if the C++ compiler below C++14.

typedef struct {
    float x;
    float y;
} mcFloatComplex;

typedef struct {
    double x;
    double y;
} mcDoubleComplex;

using mcComplex = mcFloatComplex;

#else
// Full support of complex arithmetic and classes.

#include <math.h>

#include <complex>
#include <ostream>
#include <type_traits>

#if defined(__MACA__)
/*! \brief mcsparse_complex_num is a structure which represents a complex number
 *         with precision T.
 */
template <typename T>
class mcsparse_complex_num {
 public:
    T x;  // The real part of the number.
    T y;  // The imaginary part of the number.

    // Internal real absolute function, to be sure we're on both device and host
    static __device__ __host__ T abs(T x) { return x < 0 ? -x : x; }

    static __device__ __host__ float sqrt(float x) { return ::sqrtf(x); }

    static __device__ __host__ double sqrt(double x) { return ::sqrt(x); }

 public:
    // We do not initialize the members x or y by default, to ensure that it can
    // be used in __shared__ and that it is a trivial class compatible with C.
    __device__ __host__ mcsparse_complex_num() = default;
    __device__ __host__ mcsparse_complex_num(const mcsparse_complex_num&) = default;
    __device__ __host__ mcsparse_complex_num(mcsparse_complex_num&&) = default;
    __device__ __host__ mcsparse_complex_num& operator=(const mcsparse_complex_num& rhs) & {
        this->x = rhs.x;
        this->y = rhs.y;
        return *this;
    }
    __device__ __host__ mcsparse_complex_num& operator=(mcsparse_complex_num&& rhs) & {
        this->x = rhs.x;
        this->y = rhs.y;
        return *this;
    }
    __device__ __host__ ~mcsparse_complex_num() = default;
    using value_type = T;

    // Constructor
    __device__ __host__ constexpr mcsparse_complex_num(T r, T i) : x{r}, y{i} {}

    // Conversion from real
    __device__ __host__ mcsparse_complex_num(T r) : x{r}, y{0} {}

    // Conversion from std::complex<T>
    __device__ __host__ constexpr mcsparse_complex_num(const std::complex<T>& z) : x{z.real()}, y{z.imag()} {}

    // Conversion to std::complex<T>
    __device__ __host__ constexpr operator std::complex<T>() const { return {x, y}; }

    // Conversion from different complex (explicit)
    template <typename U, std::enable_if_t<std::is_constructible<T, U>{}, int> = 0>
    __device__ __host__ explicit constexpr mcsparse_complex_num(const mcsparse_complex_num<U>& z)
        : x(z.real()), y(z.imag()) {}

    // Conversion to bool
    __device__ __host__ constexpr explicit operator bool() const { return x || y; }

    // Setters like C++20
    __device__ __host__ constexpr void real(T r) { x = r; }

    __device__ __host__ constexpr void imag(T i) { y = i; }

    // Accessors
    friend __device__ __host__ T std::real(const mcsparse_complex_num& z);
    friend __device__ __host__ T std::imag(const mcsparse_complex_num& z);

    __device__ __host__ constexpr T real() const { return x; }

    __device__ __host__ constexpr T imag() const { return y; }

    // stream output
    friend auto& operator<<(std::ostream& out, const mcsparse_complex_num& z) {
        return out << '(' << z.x << ',' << z.y << ')';
    }

    // complex-real operations
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ auto& operator+=(const U& rhs) {
        return (x += T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ auto& operator-=(const U& rhs) {
        return (x -= T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ auto& operator*=(const U& rhs) {
        return (x *= rhs), (y *= T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ auto& operator/=(const U& rhs) {
        return (x /= T(rhs)), (y /= T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ constexpr bool operator==(const U& rhs) const {
        return x == T(rhs) && y == 0;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ constexpr bool operator!=(const U& rhs) const {
        return !(*this == rhs);
    }

    // Increment and decrement
    __device__ __host__ auto& operator++() { return ++x, *this; }

    __device__ __host__ mcsparse_complex_num operator++(int) { return {x++, y}; }

    __device__ __host__ auto& operator--() { return --x, *this; }

    __device__ __host__ mcsparse_complex_num operator--(int) { return {x--, y}; }

    // Unary operations
    __device__ __host__ mcsparse_complex_num operator-() const { return {-x, -y}; }

    __device__ __host__ mcsparse_complex_num operator+() const { return *this; }

    friend __device__ __host__ T asum(const mcsparse_complex_num& z) { return abs(z.x) + abs(z.y); }

    friend __device__ __host__ mcsparse_complex_num std::conj(const mcsparse_complex_num& z);
    friend __device__ __host__ T std::norm(const mcsparse_complex_num& z);
    friend __device__ __host__ T std::abs(const mcsparse_complex_num<T>& z);

    // in-place complex-complex operations
    __device__ __host__ auto& operator*=(const mcsparse_complex_num& rhs) {
        return *this = {x * rhs.x - y * rhs.y, y * rhs.x + x * rhs.y};
    }

    __device__ __host__ auto& operator+=(const mcsparse_complex_num& rhs) { return *this = {x + rhs.x, y + rhs.y}; }

    __device__ __host__ auto& operator-=(const mcsparse_complex_num& rhs) { return *this = {x - rhs.x, y - rhs.y}; }

    __device__ __host__ auto& operator/=(const mcsparse_complex_num& rhs) {
        // Form of Robert L. Smith's Algorithm 116
        if (abs(rhs.x) > abs(rhs.y)) {
            T ratio = rhs.y / rhs.x;
            T scale = 1 / (rhs.x + rhs.y * ratio);
            *this = {(x + y * ratio) * scale, (y - x * ratio) * scale};
        } else {
            T ratio = rhs.x / rhs.y;
            T scale = 1 / (rhs.x * ratio + rhs.y);
            *this = {(y + x * ratio) * scale, (y * ratio - x) * scale};
        }
        return *this;
    }

    // out-of-place complex-complex operations
    __device__ __host__ auto operator+(const mcsparse_complex_num& rhs) const {
        auto lhs = *this;
        return lhs += rhs;
    }

    __device__ __host__ auto operator-(const mcsparse_complex_num& rhs) const {
        auto lhs = *this;
        return lhs -= rhs;
    }

    __device__ __host__ auto operator*(const mcsparse_complex_num& rhs) const {
        auto lhs = *this;
        return lhs *= rhs;
    }

    __device__ __host__ auto operator/(const mcsparse_complex_num& rhs) const {
        auto lhs = *this;
        return lhs /= rhs;
    }

    __device__ __host__ constexpr bool operator==(const mcsparse_complex_num& rhs) const {
        return x == rhs.x && y == rhs.y;
    }

    __device__ __host__ constexpr bool operator!=(const mcsparse_complex_num& rhs) const { return !(*this == rhs); }

    // real-complex operations (complex-real is handled above)
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ mcsparse_complex_num operator+(const U& lhs, const mcsparse_complex_num& rhs) {
        return {T(lhs) + rhs.x, rhs.y};
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ mcsparse_complex_num operator-(const U& lhs, const mcsparse_complex_num& rhs) {
        return {T(lhs) - rhs.x, -rhs.y};
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ mcsparse_complex_num operator*(const U& lhs, const mcsparse_complex_num& rhs) {
        return {T(lhs) * rhs.x, T(lhs) * rhs.y};
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ mcsparse_complex_num operator/(const U& lhs, const mcsparse_complex_num& rhs) {
        // Form of Robert L. Smith's Algorithm 116
        // https://dl.acm.org/doi/10.1145/368637.368661
        if (abs(rhs.x) > abs(rhs.y)) {
            T ratio = rhs.y / rhs.x;
            T scale = T(lhs) / (rhs.x + rhs.y * ratio);
            return {scale, -scale * ratio};
        } else {
            T ratio = rhs.x / rhs.y;
            T scale = T(lhs) / (rhs.x * ratio + rhs.y);
            return {ratio * scale, -scale};
        }
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ constexpr bool operator==(const U& lhs, const mcsparse_complex_num& rhs) {
        return T(lhs) == rhs.x && 0 == rhs.y;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ constexpr bool operator!=(const U& lhs, const mcsparse_complex_num& rhs) {
        return !(lhs == rhs);
    }
};

// Inject standard functions into namespace std
namespace std {
template <typename T>
__device__ __host__ constexpr T real(const mcsparse_complex_num<T>& z) {
    return z.x;
}

template <typename T>
__device__ __host__ constexpr T imag(const mcsparse_complex_num<T>& z) {
    return z.y;
}

template <typename T>
__device__ __host__ constexpr mcsparse_complex_num<T> conj(const mcsparse_complex_num<T>& z) {
    return {z.x, -z.y};
}

template <typename T>
__device__ __host__ inline T norm(const mcsparse_complex_num<T>& z) {
    return (z.x * z.x) + (z.y * z.y);
}

template <typename T>
__device__ __host__ inline T abs(const mcsparse_complex_num<T>& z) {
    T tr = mcsparse_complex_num<T>::abs(z.x), ti = mcsparse_complex_num<T>::abs(z.y);
    // clang-format off
        return tr > ti ? (ti /= tr, tr * mcsparse_complex_num<T>::sqrt(ti * ti + 1))
                       : ti ? (tr /= ti, ti * mcsparse_complex_num<T>::sqrt(tr * tr + 1)) : 0;
    // clang-format on
}
}  // namespace std

// Test for C compatibility
template <typename T>
class mcsparse_complex_num_check {
    static_assert(std::is_standard_layout<mcsparse_complex_num<T> >{},
                  "mcsparse_complex_num<T> is not a standard layout type, and thus is incompatible with C.");

    // Internal use for C++ only, remove the trivial type restriction.
    // static_assert(std::is_trivial<mcsparse_complex_num<T> >{},
    //               "mcsparse_complex_num<T> is not a trivial type, and thus is incompatible with C.");

    static_assert(sizeof(mcsparse_complex_num<T>) == 2 * sizeof(T),
                  "mcsparse_complex_num<T> is not the correct size, and thus is incompatible with C.");
};

template class mcsparse_complex_num_check<float>;
template class mcsparse_complex_num_check<double>;

// mcsparse complex data types
using mcFloatComplex = mcsparse_complex_num<float>;
using mcDoubleComplex = mcsparse_complex_num<double>;
using mcComplex = mcFloatComplex;

/*! \brief is_complex<T> returns true iff T is complex */
template <typename T>
inline constexpr bool is_complex = false;

template <>
inline constexpr bool is_complex<mcFloatComplex> = true;

template <>
inline constexpr bool is_complex<mcDoubleComplex> = true;

//!
//! @brief Struct to define pair of value and index.
//!
template <typename T>
struct mcsparse_index_value_t {
    //! @brief Important: index must come first, so that mcsparse_index_value_t* can be cast to
    //! mcsparse_int*
    int index;
    //! @brief The value.
    T value;
};

// Get base types from complex types.
template <typename T, typename = void>
struct mcsparse_real_t_impl {
    using type = T;
};

template <typename T>
struct mcsparse_real_t_impl<T, std::enable_if_t<is_complex<T> > > {
    using type = decltype(std::real(T{}));
};

template <typename T>
struct mcsparse_real_t_impl<std::complex<T> > {
    using type = T;
};

template <typename T>
using real_t = typename mcsparse_real_t_impl<T>::type;

#else
// enable for gnu
/*! \brief mcsparse_complex_num is a structure which represents a complex number
 *         with precision T.
 */
template <typename T>
class mcsparse_complex_num {
 public:
    T x;  // The real part of the number.
    T y;  // The imaginary part of the number.

    // Internal real absolute function, to be sure we're on both device and host
    static T abs(T x) { return x < 0 ? -x : x; }

    static float sqrt(float x) { return ::sqrtf(x); }

    static double sqrt(double x) { return ::sqrt(x); }

 public:
    // We do not initialize the members x or y by default, to ensure that it can
    // be used in __shared__ and that it is a trivial class compatible with C.
    mcsparse_complex_num() = default;
    mcsparse_complex_num(const mcsparse_complex_num&) = default;
    mcsparse_complex_num(mcsparse_complex_num&&) = default;
    mcsparse_complex_num& operator=(const mcsparse_complex_num& rhs) & {
        this->x = rhs.x;
        this->y = rhs.y;
        return *this;
    }
    mcsparse_complex_num& operator=(mcsparse_complex_num&& rhs) & {
        this->x = rhs.x;
        this->y = rhs.y;
        return *this;
    }
    ~mcsparse_complex_num() = default;
    using value_type = T;

    // Constructor
    constexpr mcsparse_complex_num(T r, T i) : x{r}, y{i} {}

    // Conversion from real
    mcsparse_complex_num(T r) : x{r}, y{0} {}

    // Conversion from std::complex<T>
    constexpr mcsparse_complex_num(const std::complex<T>& z) : x{z.real()}, y{z.imag()} {}

    // Conversion to std::complex<T>
    constexpr operator std::complex<T>() const { return {x, y}; }

    // Conversion from different complex (explicit)
    template <typename U, std::enable_if_t<std::is_constructible<T, U>{}, int> = 0>
    explicit constexpr mcsparse_complex_num(const mcsparse_complex_num<U>& z) : x(z.real()), y(z.imag()) {}

    // Conversion to bool
    constexpr explicit operator bool() const { return x || y; }

    // Setters like C++20
    constexpr void real(T r) { x = r; }

    constexpr void imag(T i) { y = i; }

    // Accessors
    friend T std::real(const mcsparse_complex_num& z);
    friend T std::imag(const mcsparse_complex_num& z);

    constexpr T real() const { return x; }

    constexpr T imag() const { return y; }

    // stream output
    friend auto& operator<<(std::ostream& out, const mcsparse_complex_num& z) {
        return out << '(' << z.x << ',' << z.y << ')';
    }

    // complex-real operations
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    auto& operator+=(const U& rhs) {
        return (x += T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    auto& operator-=(const U& rhs) {
        return (x -= T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    auto& operator*=(const U& rhs) {
        return (x *= rhs), (y *= T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    auto& operator/=(const U& rhs) {
        return (x /= T(rhs)), (y /= T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    constexpr bool operator==(const U& rhs) const {
        return x == T(rhs) && y == 0;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    constexpr bool operator!=(const U& rhs) const {
        return !(*this == rhs);
    }

    // Increment and decrement
    auto& operator++() { return ++x, *this; }

    mcsparse_complex_num operator++(int) { return {x++, y}; }

    auto& operator--() { return --x, *this; }

    mcsparse_complex_num operator--(int) { return {x--, y}; }

    // Unary operations
    mcsparse_complex_num operator-() const { return {-x, -y}; }

    mcsparse_complex_num operator+() const { return *this; }

    friend T asum(const mcsparse_complex_num& z) { return abs(z.x) + abs(z.y); }

    friend mcsparse_complex_num std::conj(const mcsparse_complex_num& z);
    friend T std::norm(const mcsparse_complex_num& z);
    friend T std::abs(const mcsparse_complex_num<T>& z);

    // in-place complex-complex operations
    auto& operator*=(const mcsparse_complex_num& rhs) { return *this = {x * rhs.x - y * rhs.y, y * rhs.x + x * rhs.y}; }

    auto& operator+=(const mcsparse_complex_num& rhs) { return *this = {x + rhs.x, y + rhs.y}; }

    auto& operator-=(const mcsparse_complex_num& rhs) { return *this = {x - rhs.x, y - rhs.y}; }

    auto& operator/=(const mcsparse_complex_num& rhs) {
        // Form of Robert L. Smith's Algorithm 116
        if (abs(rhs.x) > abs(rhs.y)) {
            T ratio = rhs.y / rhs.x;
            T scale = 1 / (rhs.x + rhs.y * ratio);
            *this = {(x + y * ratio) * scale, (y - x * ratio) * scale};
        } else {
            T ratio = rhs.x / rhs.y;
            T scale = 1 / (rhs.x * ratio + rhs.y);
            *this = {(y + x * ratio) * scale, (y * ratio - x) * scale};
        }
        return *this;
    }

    // out-of-place complex-complex operations
    auto operator+(const mcsparse_complex_num& rhs) const {
        auto lhs = *this;
        return lhs += rhs;
    }

    auto operator-(const mcsparse_complex_num& rhs) const {
        auto lhs = *this;
        return lhs -= rhs;
    }

    auto operator*(const mcsparse_complex_num& rhs) const {
        auto lhs = *this;
        return lhs *= rhs;
    }

    auto operator/(const mcsparse_complex_num& rhs) const {
        auto lhs = *this;
        return lhs /= rhs;
    }

    constexpr bool operator==(const mcsparse_complex_num& rhs) const { return x == rhs.x && y == rhs.y; }

    constexpr bool operator!=(const mcsparse_complex_num& rhs) const { return !(*this == rhs); }

    // real-complex operations (complex-real is handled above)
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend mcsparse_complex_num operator+(const U& lhs, const mcsparse_complex_num& rhs) {
        return {T(lhs) + rhs.x, rhs.y};
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend mcsparse_complex_num operator-(const U& lhs, const mcsparse_complex_num& rhs) {
        return {T(lhs) - rhs.x, -rhs.y};
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend mcsparse_complex_num operator*(const U& lhs, const mcsparse_complex_num& rhs) {
        return {T(lhs) * rhs.x, T(lhs) * rhs.y};
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend mcsparse_complex_num operator/(const U& lhs, const mcsparse_complex_num& rhs) {
        // Form of Robert L. Smith's Algorithm 116
        // https://dl.acm.org/doi/10.1145/368637.368661
        if (abs(rhs.x) > abs(rhs.y)) {
            T ratio = rhs.y / rhs.x;
            T scale = T(lhs) / (rhs.x + rhs.y * ratio);
            return {scale, -scale * ratio};
        } else {
            T ratio = rhs.x / rhs.y;
            T scale = T(lhs) / (rhs.x * ratio + rhs.y);
            return {ratio * scale, -scale};
        }
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend constexpr bool operator==(const U& lhs, const mcsparse_complex_num& rhs) {
        return T(lhs) == rhs.x && 0 == rhs.y;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend constexpr bool operator!=(const U& lhs, const mcsparse_complex_num& rhs) {
        return !(lhs == rhs);
    }
};

// Inject standard functions into namespace std
namespace std {
template <typename T>
constexpr T real(const mcsparse_complex_num<T>& z) {
    return z.x;
}

template <typename T>
constexpr T imag(const mcsparse_complex_num<T>& z) {
    return z.y;
}

template <typename T>
constexpr mcsparse_complex_num<T> conj(const mcsparse_complex_num<T>& z) {
    return {z.x, -z.y};
}

template <typename T>
inline T norm(const mcsparse_complex_num<T>& z) {
    return (z.x * z.x) + (z.y * z.y);
}

template <typename T>
inline T abs(const mcsparse_complex_num<T>& z) {
    T tr = mcsparse_complex_num<T>::abs(z.x), ti = mcsparse_complex_num<T>::abs(z.y);
    // clang-format off
        return tr > ti ? (ti /= tr, tr * mcsparse_complex_num<T>::sqrt(ti * ti + 1))
                       : ti ? (tr /= ti, ti * mcsparse_complex_num<T>::sqrt(tr * tr + 1)) : 0;
    // clang-format on
}
}  // namespace std

// Test for C compatibility
template <typename T>
class mcsparse_complex_num_check {
    static_assert(std::is_standard_layout<mcsparse_complex_num<T> >{},
                  "mcsparse_complex_num<T> is not a standard layout type, and thus is incompatible with C.");

    // Internal use for C++ only, remove the trivial type restriction.
    // static_assert(std::is_trivial<mcsparse_complex_num<T> >{},
    //               "mcsparse_complex_num<T> is not a trivial type, and thus is incompatible with C.");

    static_assert(sizeof(mcsparse_complex_num<T>) == 2 * sizeof(T),
                  "mcsparse_complex_num<T> is not the correct size, and thus is incompatible with C.");
};

template class mcsparse_complex_num_check<float>;
template class mcsparse_complex_num_check<double>;

// mcsparse complex data types
using mcFloatComplex = mcsparse_complex_num<float>;
using mcDoubleComplex = mcsparse_complex_num<double>;
using mcComplex = mcFloatComplex;

/*! \brief is_complex<T> returns true iff T is complex */
template <typename T>
inline constexpr bool is_complex = false;

template <>
inline constexpr bool is_complex<mcFloatComplex> = true;

template <>
inline constexpr bool is_complex<mcDoubleComplex> = true;

//!
//! @brief Struct to define pair of value and index.
//!
template <typename T>
struct mcsparse_index_value_t {
    //! @brief Important: index must come first, so that mcsparse_index_value_t* can be cast to
    //! mcsparse_int*
    int index;
    //! @brief The value.
    T value;
};

// Get base types from complex types.
template <typename T, typename = void>
struct mcsparse_real_t_impl {
    using type = T;
};

template <typename T>
struct mcsparse_real_t_impl<T, std::enable_if_t<is_complex<T> > > {
    using type = decltype(std::real(T{}));
};

template <typename T>
struct mcsparse_real_t_impl<std::complex<T> > {
    using type = T;
};

template <typename T>
using real_t = typename mcsparse_real_t_impl<T>::type;

#endif  // __clang__

#endif  // __cplusplus < 201402L

#endif  // COMMON_MCSP_COMPLEX_H_
