#include "tensor/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#ifdef USE_BLAS
    #ifdef USE_ACCELERATE
        #include <Accelerate/Accelerate.h>
    #else
        #include <cblas.h>
    #endif
#endif

#define OMP_THRESHOLD 500000
#define SIMD_THRESHOLD 32

struct Tensor {
    float* data;           // Pointer to data (may be offset into data_owner)
    float* data_owner;     // Pointer to start of owned memory (NULL if view)
    usize* dims;           // Shape of the tensor
    usize ndims;           // Number of dimensions
    usize length;          // Total accessible elements
    usize* stride;         // Stride for each dimension
    usize offset;          // Offset from data_owner
    usize* ref_count;      // Reference count for shared data (shared across views)
};


// Fast path for contiguous tensors - direct linear indexing
#define TENSOR_LOOP_CONTIGUOUS(size, body) do { \
    if (size > OMP_THRESHOLD) { \
        _Pragma("omp parallel for simd") \
        for (usize _i = 0; _i < size; _i++) { body; } \
    } else if (size > SIMD_THRESHOLD) { \
        _Pragma("omp simd") \
        for (usize _i = 0; _i < size; _i++) { body; } \
    } else { \
        for (usize _i = 0; _i < size; _i++) { body; } \
    } \
} while(0)

// Compute linear offset from multi-dimensional index using strides
static inline usize compute_offset(usize* idx, usize* stride, usize ndims) {
    usize offset = 0;
    for (usize d = 0; d < ndims; d++) {
        offset += idx[d] * stride[d];
    }
    return offset;
}

// Increment multi-dimensional index (like an odometer)
static inline void increment_index(usize* idx, usize* dims, usize ndims) {
    for (isize d = ndims - 1; d >= 0; d--) {
        idx[d]++;
        if (idx[d] < dims[d]) {
            return;
        }
        idx[d] = 0;
    }
}

// General loop that works for both contiguous and non-contiguous tensors
#define TENSOR_LOOP(tensor, body) do { \
    if (TensorIsContiguous(tensor)) { \
        usize size = (tensor)->length; \
        float* restrict data = (tensor)->data; \
        if (size > OMP_THRESHOLD) { \
            _Pragma("omp parallel for simd") \
            for (usize _i = 0; _i < size; _i++) { body; } \
        } else if (size > SIMD_THRESHOLD) { \
            _Pragma("omp simd") \
            for (usize _i = 0; _i < size; _i++) { body; } \
        } else { \
            for (usize _i = 0; _i < size; _i++) { body; } \
        } \
    } else { \
        usize* _idx = calloc((tensor)->ndims, sizeof(usize)); \
        for (usize _i = 0; _i < (tensor)->length; _i++) { \
            usize _offset = compute_offset(_idx, (tensor)->stride, (tensor)->ndims); \
            body; \
            increment_index(_idx, (tensor)->dims, (tensor)->ndims); \
        } \
        free(_idx); \
    } \
} while(0)

#define CHECK_NULL(ptr, msg, t) \
    if (!ptr) { \
        fprintf(stderr, msg); \
        if (t) { \
            TensorFree(t); \
        } \
        abort(); \
    } 

#define CHECK_N_DIMS(ndims, msg) \
    if (ndims == 0) { \
        fprintf(stderr, msg); \
        abort(); \
    }

#define CHECK_EQUAL_SHAPE(a, b, fn_name) do { \
    if (!TensorShapeEqual(a, b)) { \
        fprintf(stderr, "%s: Cannot perform operation between tensor of shape [", fn_name); \
        for (usize _i = 0; _i < TensorNDims(a); _i++) { \
            fprintf(stderr, "%zu", TensorDim(a, _i)); \
            if (_i < TensorNDims(a) - 1) fprintf(stderr, ", "); \
        } \
        fprintf(stderr, "] and shape ["); \
        for (usize _i = 0; _i < TensorNDims(b); _i++) { \
            fprintf(stderr, "%zu", TensorDim(b, _i)); \
            if (_i < TensorNDims(b) - 1) fprintf(stderr, ", "); \
        } \
        fprintf(stderr, "]\n"); \
        abort(); \
    } \
} while(0)

#define DEFINE_UNARY_OP(name, op)                                   \
Tensor* name(Tensor* a) {                                           \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                      \
    Tensor* t = TensorNew(a->dims, a->ndims);                       \
    if (TensorIsContiguous(a)) {                                    \
        float* restrict t_data = t->data;                           \
        float* restrict a_data = a->data;                           \
        TENSOR_LOOP_CONTIGUOUS(a->length, t_data[_i] = op(a_data[_i])); \
    } else {                                                        \
        usize* idx = calloc(a->ndims, sizeof(usize));              \
        for (usize i = 0; i < a->length; i++) {                     \
            float val = TensorGet(a, idx);                          \
            t->data[i] = op(val);                                   \
            increment_index(idx, a->dims, a->ndims);                \
        }                                                           \
        free(idx);                                                  \
    }                                                               \
    return t;                                                       \
}

#define DEFINE_UNARY_OP_(name, op)                                  \
void name(Tensor* a) {                                              \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                      \
    if (TensorIsContiguous(a)) {                                    \
        float* restrict a_data = a->data;                           \
        TENSOR_LOOP_CONTIGUOUS(a->length, a_data[_i] = op(a_data[_i])); \
    } else {                                                        \
        usize* idx = calloc(a->ndims, sizeof(usize));              \
        for (usize i = 0; i < a->length; i++) {                     \
            float val = TensorGet(a, idx);                          \
            TensorSet(a, idx, op(val));                             \
            increment_index(idx, a->dims, a->ndims);                \
        }                                                           \
        free(idx);                                                  \
    }                                                               \
}

#define OP_NEG(x)       (-(x))
#define OP_ABS(x)       ((x) < 0 ? -(x) : (x))
#define OP_SIGN(x)      (((x) > 0) - ((x) < 0))
#define OP_SQUARE(x)    ((x) * (x))
#define OP_RECIP(x)     (1.0f / (x))
#define OP_RELU(x)      ((x) > 0 ? (x) : 0)
#define OP_SIGMOID(x)   (1.0f / (1.0f + expf(-(x))))
#define OP_ISNAN(x)     (isnan(x) ? 1.0f : 0.0f)
#define OP_ISINF(x)     (isinf(x) ? 1.0f : 0.0f)
#define OP_FLOOR(x)     floorf(x)
#define OP_CEIL(x)      ceilf(x)
#define OP_ROUND(x)     roundf(x)

DEFINE_UNARY_OP(TensorNeg,        OP_NEG)
DEFINE_UNARY_OP(TensorAbs,        OP_ABS)
DEFINE_UNARY_OP(TensorSign,       OP_SIGN)
DEFINE_UNARY_OP(TensorSquare,     OP_SQUARE)
DEFINE_UNARY_OP(TensorReciprocal,      OP_RECIP)
DEFINE_UNARY_OP(TensorExp,        expf)
DEFINE_UNARY_OP(TensorLog,        logf)
DEFINE_UNARY_OP(TensorSqrt,       sqrtf)
DEFINE_UNARY_OP(TensorSin,        sinf)
DEFINE_UNARY_OP(TensorCos,        cosf)
DEFINE_UNARY_OP(TensorTan,        tanf)
DEFINE_UNARY_OP(TensorAsin,       asinf)
DEFINE_UNARY_OP(TensorAcos,       acosf)
DEFINE_UNARY_OP(TensorAtan,       atanf)
DEFINE_UNARY_OP(TensorAtanh,      atanhf)
DEFINE_UNARY_OP(TensorSinh,       sinhf)
DEFINE_UNARY_OP(TensorCosh,       coshf)
DEFINE_UNARY_OP(TensorTanh,       tanhf)
DEFINE_UNARY_OP(TensorFloor,      OP_FLOOR)
DEFINE_UNARY_OP(TensorCeil,       OP_CEIL)
DEFINE_UNARY_OP(TensorRound,      OP_ROUND)
DEFINE_UNARY_OP(TensorRelu,       OP_RELU)
DEFINE_UNARY_OP(TensorSigmoid,    OP_SIGMOID)
DEFINE_UNARY_OP(TensorIsNan,      OP_ISNAN)
DEFINE_UNARY_OP(TensorIsInf,      OP_ISINF)

DEFINE_UNARY_OP_(TensorNeg_,      OP_NEG)
DEFINE_UNARY_OP_(TensorAbs_,      OP_ABS)
DEFINE_UNARY_OP_(TensorExp_,      expf)
DEFINE_UNARY_OP_(TensorLog_,      logf)
DEFINE_UNARY_OP_(TensorSqrt_,     sqrtf)
DEFINE_UNARY_OP_(TensorSin_,      sinf)
DEFINE_UNARY_OP_(TensorCos_,      cosf)
DEFINE_UNARY_OP_(TensorAsin_,     asinf)
DEFINE_UNARY_OP_(TensorAcos_,     acosf)
DEFINE_UNARY_OP_(TensorAtan_,     atanf)
DEFINE_UNARY_OP_(TensorAtanh_,    atanhf)
DEFINE_UNARY_OP_(TensorSinh_,     sinhf)
DEFINE_UNARY_OP_(TensorTan_,      tanf)
DEFINE_UNARY_OP_(TensorSign_,     OP_SIGN)
DEFINE_UNARY_OP_(TensorReciprocal_, OP_RECIP)
DEFINE_UNARY_OP_(TensorSquare_,   OP_SQUARE)
DEFINE_UNARY_OP_(TensorCosh_,     coshf)
DEFINE_UNARY_OP_(TensorTanh_,     tanhf)
DEFINE_UNARY_OP_(TensorFloor_,    OP_FLOOR)
DEFINE_UNARY_OP_(TensorCeil_,     OP_CEIL)
DEFINE_UNARY_OP_(TensorRound_,    OP_ROUND)
DEFINE_UNARY_OP_(TensorRelu_,     OP_RELU)
DEFINE_UNARY_OP_(TensorSigmoid_,  OP_SIGMOID)

#define DEFINE_BINARY_OP(name, op_contiguous, op_strided)           \
Tensor* name(Tensor* a, Tensor* b) {                                \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                      \
    CHECK_NULL(b, #name ": b is NULL\n", NULL);                      \
    CHECK_EQUAL_SHAPE(a, b, #name);                                   \
    Tensor* t = TensorNew(a->dims, a->ndims);                       \
    if (TensorIsContiguous(a) && TensorIsContiguous(b)) {           \
        float* restrict t_data = t->data;                           \
        float* restrict a_data = a->data;                           \
        float* restrict b_data = b->data;                           \
        TENSOR_LOOP_CONTIGUOUS(a->length, t_data[_i] = op_contiguous); \
    } else {                                                        \
        usize* idx = calloc(a->ndims, sizeof(usize));              \
        for (usize i = 0; i < a->length; i++) {                     \
            float a_val = TensorGet(a, idx);                        \
            float b_val = TensorGet(b, idx);                        \
            t->data[i] = op_strided;                                \
            increment_index(idx, a->dims, a->ndims);                \
        }                                                           \
        free(idx);                                                  \
    }                                                               \
    return t;                                                       \
}

#define DEFINE_BINARY_OP_(name, op_contiguous, op_strided)          \
void name(Tensor* a, Tensor* b) {                                   \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                      \
    CHECK_NULL(b, #name ": b is NULL\n", NULL);                      \
    CHECK_EQUAL_SHAPE(a, b, #name);                                   \
    if (TensorIsContiguous(a) && TensorIsContiguous(b)) {           \
        float* restrict a_data = a->data;                           \
        float* restrict b_data = b->data;                           \
        TENSOR_LOOP_CONTIGUOUS(a->length, op_contiguous);           \
    } else {                                                        \
        usize* idx = calloc(a->ndims, sizeof(usize));              \
        for (usize i = 0; i < a->length; i++) {                     \
            float a_val = TensorGet(a, idx);                        \
            float b_val = TensorGet(b, idx);                        \
            TensorSet(a, idx, op_strided);                          \
            increment_index(idx, a->dims, a->ndims);                \
        }                                                           \
        free(idx);                                                  \
    }                                                               \
}

DEFINE_BINARY_OP(TensorAdd,   a_data[_i] + b_data[_i],   a_val + b_val)
DEFINE_BINARY_OP(TensorSub,   a_data[_i] - b_data[_i],   a_val - b_val)
DEFINE_BINARY_OP(TensorMul,   a_data[_i] * b_data[_i],   a_val * b_val)
DEFINE_BINARY_OP(TensorDiv,   a_data[_i] / b_data[_i],   a_val / b_val)
DEFINE_BINARY_OP(TensorPowT,  powf(a_data[_i], b_data[_i]),  powf(a_val, b_val))
DEFINE_BINARY_OP(TensorMod,   fmodf(a_data[_i], b_data[_i]), fmodf(a_val, b_val))
DEFINE_BINARY_OP(TensorAtan2, atan2f(a_data[_i], b_data[_i]), atan2f(a_val, b_val))
DEFINE_BINARY_OP(TensorMin2,  fminf(a_data[_i], b_data[_i]), fminf(a_val, b_val))
DEFINE_BINARY_OP(TensorMax2,  fmaxf(a_data[_i], b_data[_i]), fmaxf(a_val, b_val))

DEFINE_BINARY_OP(TensorEq,    (a_data[_i] == b_data[_i]) ? 1.0f : 0.0f,  (a_val == b_val) ? 1.0f : 0.0f)
DEFINE_BINARY_OP(TensorNe,    (a_data[_i] != b_data[_i]) ? 1.0f : 0.0f,  (a_val != b_val) ? 1.0f : 0.0f)
DEFINE_BINARY_OP(TensorGt,    (a_data[_i] >  b_data[_i]) ? 1.0f : 0.0f,  (a_val >  b_val) ? 1.0f : 0.0f)
DEFINE_BINARY_OP(TensorLt,    (a_data[_i] <  b_data[_i]) ? 1.0f : 0.0f,  (a_val <  b_val) ? 1.0f : 0.0f)
DEFINE_BINARY_OP(TensorGe,    (a_data[_i] >= b_data[_i]) ? 1.0f : 0.0f,  (a_val >= b_val) ? 1.0f : 0.0f)
DEFINE_BINARY_OP(TensorLe,    (a_data[_i] <= b_data[_i]) ? 1.0f : 0.0f,  (a_val <= b_val) ? 1.0f : 0.0f)

DEFINE_BINARY_OP(TensorAnd,   (a_data[_i] != 0 && b_data[_i] != 0) ? 1.0f : 0.0f,  (a_val != 0 && b_val != 0) ? 1.0f : 0.0f)
DEFINE_BINARY_OP(TensorOr,    (a_data[_i] != 0 || b_data[_i] != 0) ? 1.0f : 0.0f,  (a_val != 0 || b_val != 0) ? 1.0f : 0.0f)
DEFINE_BINARY_OP(TensorXor,   ((a_data[_i] != 0) != (b_data[_i] != 0)) ? 1.0f : 0.0f,  ((a_val != 0) != (b_val != 0)) ? 1.0f : 0.0f)

DEFINE_BINARY_OP_(TensorAdd_, a_data[_i] += b_data[_i],  a_val + b_val)
DEFINE_BINARY_OP_(TensorSub_, a_data[_i] -= b_data[_i],  a_val - b_val)
DEFINE_BINARY_OP_(TensorMul_, a_data[_i] *= b_data[_i],  a_val * b_val)
DEFINE_BINARY_OP_(TensorDiv_, a_data[_i] /= b_data[_i],  a_val / b_val)


#define DEFINE_SCALAR_OP(name, op_contiguous, op_strided)           \
Tensor* name(Tensor* a, float s) {                                  \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                      \
    Tensor* t = TensorNew(a->dims, a->ndims);                       \
    if (TensorIsContiguous(a)) {                                    \
        float* restrict t_data = t->data;                           \
        float* restrict a_data = a->data;                           \
        TENSOR_LOOP_CONTIGUOUS(a->length, t_data[_i] = op_contiguous); \
    } else {                                                        \
        usize* idx = calloc(a->ndims, sizeof(usize));              \
        for (usize i = 0; i < a->length; i++) {                     \
            float a_val = TensorGet(a, idx);                        \
            t->data[i] = op_strided;                                \
            increment_index(idx, a->dims, a->ndims);                \
        }                                                           \
        free(idx);                                                  \
    }                                                               \
    return t;                                                       \
}

#define DEFINE_SCALAR_OP_(name, op_contiguous, op_strided)          \
void name(Tensor* a, float s) {                                     \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                      \
    if (TensorIsContiguous(a)) {                                    \
        float* restrict a_data = a->data;                           \
        TENSOR_LOOP_CONTIGUOUS(a->length, op_contiguous);           \
    } else {                                                        \
        usize* idx = calloc(a->ndims, sizeof(usize));              \
        for (usize i = 0; i < a->length; i++) {                     \
            float a_val = TensorGet(a, idx);                        \
            TensorSet(a, idx, op_strided);                          \
            increment_index(idx, a->dims, a->ndims);                \
        }                                                           \
        free(idx);                                                  \
    }                                                               \
}

DEFINE_SCALAR_OP(TensorAddScalar, a_data[_i] + s,  a_val + s)
DEFINE_SCALAR_OP(TensorSubScalar, a_data[_i] - s,  a_val - s)
DEFINE_SCALAR_OP(TensorMulScalar, a_data[_i] * s,  a_val * s)
DEFINE_SCALAR_OP(TensorDivScalar, a_data[_i] / s,  a_val / s)
DEFINE_SCALAR_OP(TensorPow,       powf(a_data[_i], s),  powf(a_val, s))

DEFINE_SCALAR_OP(TensorEqScalar,  (a_data[_i] == s) ? 1.0f : 0.0f,  (a_val == s) ? 1.0f : 0.0f)
DEFINE_SCALAR_OP(TensorNeScalar,  (a_data[_i] != s) ? 1.0f : 0.0f,  (a_val != s) ? 1.0f : 0.0f)
DEFINE_SCALAR_OP(TensorGtScalar,  (a_data[_i] >  s) ? 1.0f : 0.0f,  (a_val >  s) ? 1.0f : 0.0f)
DEFINE_SCALAR_OP(TensorLtScalar,  (a_data[_i] <  s) ? 1.0f : 0.0f,  (a_val <  s) ? 1.0f : 0.0f)
DEFINE_SCALAR_OP(TensorGeScalar,  (a_data[_i] >= s) ? 1.0f : 0.0f,  (a_val >= s) ? 1.0f : 0.0f)
DEFINE_SCALAR_OP(TensorLeScalar,  (a_data[_i] <= s) ? 1.0f : 0.0f,  (a_val <= s) ? 1.0f : 0.0f)

DEFINE_SCALAR_OP_(TensorAddScalar_, a_data[_i] += s,  a_val + s)
DEFINE_SCALAR_OP_(TensorSubScalar_, a_data[_i] -= s,  a_val - s)
DEFINE_SCALAR_OP_(TensorMulScalar_, a_data[_i] *= s,  a_val * s)
DEFINE_SCALAR_OP_(TensorDivScalar_, a_data[_i] /= s,  a_val / s)
DEFINE_SCALAR_OP_(TensorPow_,       a_data[_i] = powf(a_data[_i], s),  powf(a_val, s))

#define STRINGIZE(x) #x
#define CONCAT_PRAGMA(a, b, c) STRINGIZE(a b c)

#define DEFINE_REDUCE_ALL_IMPL(name, init, op, pragma_str)         \
float name(Tensor* a) {                                             \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                     \
    usize size = a->length;                                         \
    float result = init;                                            \
    _Pragma(pragma_str)                                             \
    for (usize i = 0; i < size; i++) {                              \
        op;                                                         \
    }                                                               \
    return result;                                                  \
}

#define DEFINE_CLAMP_OP(name, op_contiguous, op_strided)            \
Tensor* name(Tensor* a, float min, float max) {                     \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                     \
    Tensor* t = TensorNew(a->dims, a->ndims);                       \
    if (TensorIsContiguous(a)) {                                    \
        float* restrict t_data = t->data;                           \
        float* restrict a_data = a->data;                           \
        TENSOR_LOOP_CONTIGUOUS(a->length, t_data[_i] = op_contiguous); \
    } else {                                                        \
        usize* idx = calloc(a->ndims, sizeof(usize));              \
        for (usize i = 0; i < a->length; i++) {                     \
            float a_val = TensorGet(a, idx);                        \
            t->data[i] = op_strided;                                \
            increment_index(idx, a->dims, a->ndims);                \
        }                                                           \
        free(idx);                                                  \
    }                                                               \
    return t;                                                       \
}                                                                   \
void name##_(Tensor* a, float min, float max) {                     \
    CHECK_NULL(a, #name "_: a is NULL\n", NULL);                    \
    if (TensorIsContiguous(a)) {                                    \
        float* restrict a_data = a->data;                           \
        TENSOR_LOOP_CONTIGUOUS(a->length, a_data[_i] = op_contiguous); \
    } else {                                                        \
        usize* idx = calloc(a->ndims, sizeof(usize));              \
        for (usize i = 0; i < a->length; i++) {                     \
            float a_val = TensorGet(a, idx);                        \
            TensorSet(a, idx, op_strided);                          \
            increment_index(idx, a->dims, a->ndims);                \
        }                                                           \
        free(idx);                                                  \
    }                                                               \
}


#define DEFINE_REDUCE_DIM(name, op, init, post_op)                              \
Tensor* name(Tensor* a, usize dim, bool keepdim) {                              \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                                 \
    if (dim >= a->ndims) {                                                      \
        fprintf(stderr, #name ": dim %zu out of range\n", dim);                 \
        abort();                                                                \
    }                                                                           \
                                                                                \
    usize old_ndims = a->ndims;                                                 \
    usize new_ndims = keepdim ? old_ndims : old_ndims - 1;                      \
    usize* new_dims = malloc(sizeof(usize) * new_ndims);                        \
                                                                                \
    usize j = 0;                                                                \
    for (usize d = 0; d < old_ndims; d++) {                                     \
        if (d == dim) {                                                         \
            if (keepdim) new_dims[j++] = 1;                                     \
        } else {                                                                \
            new_dims[j++] = a->dims[d];                                         \
        }                                                                       \
    }                                                                           \
                                                                                \
    Tensor* result = TensorNew(new_dims, new_ndims);                            \
    usize result_size = result->length;                                         \
    usize reduce_size = a->dims[dim];                                           \
    usize reduce_stride = a->stride[dim];                                       \
                                                                                \
    usize* stride_map = malloc(sizeof(usize) * new_ndims);                      \
    j = 0;                                                                      \
    for (usize d = 0; d < old_ndims; d++) {                                     \
        if (d != dim || keepdim) {                                              \
            stride_map[j++] = (d == dim) ? 0 : a->stride[d];                    \
        }                                                                       \
    }                                                                           \
                                                                                \
    _Pragma("omp parallel for if(result_size > OMP_THRESHOLD)")                 \
    for (usize i = 0; i < result_size; i++) {                                   \
        usize base_offset = 0;                                                  \
        usize temp = i;                                                         \
        for (isize d = new_ndims - 1; d >= 0; d--) {                            \
            usize idx = temp % result->dims[d];                                 \
            temp /= result->dims[d];                                            \
            base_offset += idx * stride_map[d];                                 \
        }                                                                       \
                                                                                \
        float acc = init;                                                       \
        for (usize r = 0; r < reduce_size; r++) {                               \
            float val = a->data[base_offset + r * reduce_stride];               \
            op;                                                                 \
        }                                                                       \
        post_op;                                                                \
        result->data[i] = acc;                                                  \
    }                                                                           \
                                                                                \
    free(new_dims);                                                             \
    free(stride_map);                                                           \
    return result;                                                              \
}

DEFINE_REDUCE_DIM(TensorSumDim,  acc += val,                        0.0f,      (void)0)
DEFINE_REDUCE_DIM(TensorProdDim, acc *= val,                        1.0f,      (void)0)
DEFINE_REDUCE_DIM(TensorMaxDim,  acc = (val > acc) ? val : acc,    -INFINITY,  (void)0)
DEFINE_REDUCE_DIM(TensorMinDim,  acc = (val < acc) ? val : acc,     INFINITY,  (void)0)
DEFINE_REDUCE_DIM(TensorMeanDim, acc += val,                        0.0f,      acc /= reduce_size)


#define DEFINE_ARG_REDUCE(name, cmp)                                            \
Tensor* name(Tensor* a, usize dim, bool keepdim) {                              \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                                 \
    if (dim >= a->ndims) {                                                      \
        fprintf(stderr, #name ": dim %zu out of range\n", dim);                 \
        abort();                                                                \
    }                                                                           \
                                                                                \
    usize old_ndims = a->ndims;                                                 \
    usize new_ndims = keepdim ? old_ndims : old_ndims - 1;                      \
    usize* new_dims = malloc(sizeof(usize) * new_ndims);                        \
                                                                                \
    usize j = 0;                                                                \
    for (usize d = 0; d < old_ndims; d++) {                                     \
        if (d == dim) {                                                         \
            if (keepdim) new_dims[j++] = 1;                                     \
        } else {                                                                \
            new_dims[j++] = a->dims[d];                                         \
        }                                                                       \
    }                                                                           \
                                                                                \
    Tensor* result = TensorNew(new_dims, new_ndims);                            \
    usize result_size = result->length;                                         \
    usize reduce_size = a->dims[dim];                                           \
    usize reduce_stride = a->stride[dim];                                       \
                                                                                \
    usize* stride_map = malloc(sizeof(usize) * new_ndims);                      \
    j = 0;                                                                      \
    for (usize d = 0; d < old_ndims; d++) {                                     \
        if (d != dim || keepdim) {                                              \
            stride_map[j++] = (d == dim) ? 0 : a->stride[d];                    \
        }                                                                       \
    }                                                                           \
                                                                                \
    _Pragma("omp parallel for if(result_size > OMP_THRESHOLD)")                 \
    for (usize i = 0; i < result_size; i++) {                                   \
        usize base_offset = 0;                                                  \
        usize temp = i;                                                         \
        for (isize d = new_ndims - 1; d >= 0; d--) {                            \
            usize idx = temp % result->dims[d];                                 \
            temp /= result->dims[d];                                            \
            base_offset += idx * stride_map[d];                                 \
        }                                                                       \
                                                                                \
        float best_val = a->data[base_offset];                                  \
        usize best_idx = 0;                                                     \
        for (usize r = 1; r < reduce_size; r++) {                               \
            float val = a->data[base_offset + r * reduce_stride];               \
            if (val cmp best_val) {                                             \
                best_val = val;                                                 \
                best_idx = r;                                                   \
            }                                                                   \
        }                                                                       \
        result->data[i] = (float)best_idx;                                      \
    }                                                                           \
                                                                                \
    free(new_dims);                                                             \
    free(stride_map);                                                           \
    return result;                                                              \
}

DEFINE_ARG_REDUCE(TensorArgMaxDim, >)
DEFINE_ARG_REDUCE(TensorArgMinDim, <)

#define DEFINE_ARG_REDUCE_ALL(name, cmp)                            \
float name(Tensor* a) {                                             \
    CHECK_NULL(a, #name ": a is NULL\n", NULL);                     \
    usize size = a->length;                                         \
    float best_val = a->data[0];                                    \
    usize best_idx = 0;                                             \
    for (usize i = 1; i < size; i++) {                              \
        if (a->data[i] cmp best_val) {                              \
            best_val = a->data[i];                                  \
            best_idx = i;                                           \
        }                                                           \
    }                                                               \
    return (float)best_idx;                                         \
}


usize CalculateLength(usize* dims, usize ndims) {
    usize total = 1;
    
    for (usize i = 0; i < ndims; i++) {
        total *= dims[i];
    }

    return total;
}

// Used to initialize a stride array (in row major format).
void InitStride(usize* dims, usize ndims, usize* stride) {
    stride[ndims - 1] = 1;

    for (usize i = ndims - 1; i > 0; i--) {
        stride[i - 1] = stride[i] * dims[i];
    }
}

Tensor* TensorNew(usize* dims, usize ndims) {
    CHECK_NULL(dims, "TensorNew: dims is NULL\n", NULL);
    CHECK_N_DIMS(ndims, "TensorNew: ndims is 0\n");
    usize length = CalculateLength(dims, ndims);

    Tensor* t = malloc(sizeof(Tensor));
    CHECK_NULL(t, "TensorNew: failed to create Tensor struct\n", NULL);
    t->data = NULL; t->data_owner = NULL; t->dims = NULL;
    t->stride = NULL; t->ref_count = NULL;

    t->dims = malloc(sizeof(usize) * ndims);
    CHECK_NULL(t->dims, "TensorNew: failed to create Tensor dims array\n", t);
    memcpy(t->dims, dims, sizeof(usize) * ndims);
    t->ndims = ndims;

    t->data = calloc(length, sizeof(float));
    CHECK_NULL(t->data, "TensorNew: failed to create data array\n", t);
    t->data_owner = t->data;  // This tensor owns its data
    t->length = length;
    t->offset = 0;

    t->stride = malloc(sizeof(usize) * ndims);
    CHECK_NULL(t->stride, "TensorNew: failed to create stride array\n", t);
    InitStride(dims, ndims, t->stride);

    // Initialize reference count
    t->ref_count = malloc(sizeof(usize));
    CHECK_NULL(t->ref_count, "TensorNew: failed to create ref_count\n", t);
    *t->ref_count = 1;

    return t;
}

Tensor* TensorFrom(usize* dims, usize ndims, float* data) {
    CHECK_NULL(data, "TensorFrom: tried to create tensor from NULL data\n", NULL);
    Tensor* t = TensorNew(dims, ndims);

    memcpy(t->data, data, t->length * sizeof(float));

    return t;
}

Tensor* TensorCopy(Tensor* t) {
    CHECK_NULL(t, "TensorCopy: tried to create null tensor\n", NULL);
    Tensor* newT = TensorFrom(t->dims, t->ndims, t->data);
    return newT;
}

Tensor* TensorZero(usize* dims, usize ndims) {
    return TensorNew(dims, ndims);
}

Tensor* TensorOnes(usize* dims, usize ndims) {
    return TensorFill(dims, ndims, 1.0);
}

Tensor* TensorFill(usize* dims, usize ndims, float value) {
    Tensor* t = TensorNew(dims, ndims);
    for (usize i = 0; i < t->length; i++) {
        t->data[i] = value;
    }
    return t;
}

Tensor* TensorRandN(usize* dims, usize ndims, float mean, float std) {
    Tensor* t = TensorNew(dims, ndims);
    float* data = TensorData(t);
    usize size = TensorSize(t);
    
    for (usize i = 0; i < size; i++) {
        data[i] = gaussian(mean, std);
    }
    
    return t;
}

Tensor* TensorRandU(usize* dims, usize ndims, float low, float high) {
    Tensor* t = TensorNew(dims, ndims);
    float* data = TensorData(t);
    usize size = TensorSize(t);
    
    for (usize i = 0; i < size; i++) {
        data[i] = uniformRange(low, high);
    }
    
    return t;
}

usize TensorNDims(Tensor* t) {
    CHECK_NULL(t, "TensorNDims: Tried to get dimension of NULL tensor\n", t);
    return t->ndims;
}

usize* TensorShape(Tensor *t) {
    CHECK_NULL(t, "TensorShape: Tried to get shape of NULL tensor\n", t);
    return t->dims;
}

usize TensorSize(Tensor* t) {
    CHECK_NULL(t, "TensorSize: Tried to get size of NULL tensor\n", t);
    return t->length;
}

usize TensorDim(Tensor* t, usize d) {
    CHECK_NULL(t, "TensorDim: Tried to get dimension of NULL tensor\n", t);

    if (d >= t->ndims) {
        fprintf(stderr, "Tried to access out of bounds dimension %zu for tensor with %zu dimensions\n", d, t->ndims);
        TensorFree(t);
        abort();
    }

    return t->dims[d];
}

usize TensorStride(Tensor* t, usize d) {
    CHECK_NULL(t, "TensorStride: Tried to get stride of NULL tensor\n", t);

    if (d >= t->ndims) {
        fprintf(stderr, "Tried to access out of bounds dimension %zu for tensor with %zu dimensions\n", d, t->ndims);
        TensorFree(t);
        abort();
    }

    return t->stride[d];
}

float* TensorData(Tensor* t) {
    CHECK_NULL(t, "TensorData: NULL tensor\n", NULL);
    return t->data;
}

bool TensorIsView(Tensor* t) {
    CHECK_NULL(t, "TensorIsView: NULL tensor\n", NULL);
    // A tensor is a view if it doesn't own its data (data_owner is NULL)
    return t->data_owner == NULL;
}

bool TensorIsContiguous(Tensor* t) {
    CHECK_NULL(t, "TensorIsContiguous: NULL tensor\n", NULL);

    // A tensor is contiguous if its strides match the row-major layout
    usize expected_stride = 1;
    for (isize d = t->ndims - 1; d >= 0; d--) {
        if (t->stride[d] != expected_stride) {
            return false;
        }
        expected_stride *= t->dims[d];
    }
    return true;
}

Tensor* TensorBase(Tensor* t) {
    CHECK_NULL(t, "TensorBase: NULL tensor\n", NULL);
    // For now, just return the tensor itself
    // In a full implementation, we'd track the original base tensor
    return t;
}

Tensor* TensorView(Tensor* t) {
    CHECK_NULL(t, "TensorView: NULL tensor\n", NULL);

    Tensor* view = malloc(sizeof(Tensor));
    CHECK_NULL(view, "TensorView: failed to allocate view\n", NULL);

    // Copy dimensions
    view->dims = malloc(sizeof(usize) * t->ndims);
    CHECK_NULL(view->dims, "TensorView: failed to allocate dims\n", view);
    memcpy(view->dims, t->dims, sizeof(usize) * t->ndims);
    view->ndims = t->ndims;

    // Copy strides
    view->stride = malloc(sizeof(usize) * t->ndims);
    CHECK_NULL(view->stride, "TensorView: failed to allocate stride\n", view);
    memcpy(view->stride, t->stride, sizeof(usize) * t->ndims);

    // Share data - view doesn't own data, so data_owner is NULL
    view->data = t->data;
    view->data_owner = NULL;  // Views never own their data
    view->offset = t->offset;
    view->length = t->length;

    // Share reference count - find the actual owner's ref_count
    if (t->ref_count != NULL) {
        view->ref_count = t->ref_count;
        (*view->ref_count)++;
    } else {
        view->ref_count = NULL;
    }

    return view;
}

Tensor* TensorAsContiguous(Tensor* t) {
    CHECK_NULL(t, "TensorAsContiguous: NULL tensor\n", NULL);

    if (TensorIsContiguous(t)) {
        return TensorView(t);
    }

    // Need to create a contiguous copy
    return TensorCopy(t);
}

bool TensorShapeEqual(Tensor* a, Tensor* b) {
    CHECK_NULL(a, "TensorShapeEqual: Tried to get shape of NULL tensor\n", a);
    CHECK_NULL(b, "TensorShapeEqual: Tried to get shape of NULL tensor\n", b);

    if (a->ndims != b->ndims) {
        return false;
    }

    for (usize i = 0; i < a->ndims; i++) {
        if (a->dims[i] != b->dims[i]) {
            return false;
        }
    }
    
    return true;
}

float TensorGet(Tensor* t, usize* idx) {
    CHECK_NULL(t, "TensorGet: Tried to index NULL Tensor", NULL);
    CHECK_NULL(idx, "TensorGet: Tried to access Tensor ", t);

    usize offset = 0;
    for (usize d = 0; d < t->ndims; d++) {
        offset += idx[d] * t->stride[d];
    }

    if (offset >= t->length) {
        fprintf(stderr, "TensorGet: Tried to access out of bounds element at idx: %zu", offset);
        abort();
    }

    return t->data[offset];
}

void TensorSet(Tensor* t, usize* idx, float value) {
    CHECK_NULL(t, "TensorGet: Tried to index NULL Tensor", NULL);
    CHECK_NULL(idx, "TensorGet: Tried to access Tensor ", t);

    usize offset = 0;
    for (usize d = 0; d < t->ndims; d++) {
        offset += idx[d] * t->stride[d];
    }

    if (offset >= t->length) {
        fprintf(stderr, "TensorGet: Tried to access out of bounds element at idx: %zu", offset);
        abort();
    }
    
    t->data[offset] = value;
}

static void ResolveSlice(Slice* s, usize dim_size) {
    if (s->is_all) {
        s->start = 0;
        s->stop = dim_size;
        s->step = 1;
        return;
    }
    
    if (s->start < 0) s->start += dim_size;
    if (s->stop < 0) s->stop += dim_size;
    if (s->stop == ISIZE_MAX) s->stop = dim_size;
    
    if (s->start < 0) s->start = 0;
    if (s->start > (isize)dim_size) s->start = dim_size;
    if (s->stop < 0) s->stop = 0;
    if (s->stop > (isize)dim_size) s->stop = dim_size;
}

static void SliceCopyHelper(
    Tensor* src, Tensor* dst,
    Slice* slices,
    usize* src_idx, usize* dst_idx,
    usize depth
) {
    if (depth == src->ndims) {
        float val = TensorGet(src, src_idx);
        TensorSet(dst, dst_idx, val);
        return;
    }
    
    Slice* s = &slices[depth];
    usize dst_i = 0;
    
    for (isize i = s->start; i < s->stop; i += s->step) {
        src_idx[depth] = i;
        dst_idx[depth] = dst_i++;
        SliceCopyHelper(src, dst, slices, src_idx, dst_idx, depth + 1);
    }
}

static usize SliceLength(Slice* s) {
    if (s->stop <= s->start) return 0;
    return (s->stop - s->start + s->step - 1) / s->step;
}

Tensor* TensorSlice(Tensor* t, Slice* slices, usize nslices) {
    if (!t || !slices) return NULL;
    if (nslices != t->ndims) {
        fprintf(stderr, "TensorSlice: expected %zu slices, got %zu\n", 
                t->ndims, nslices);
        return NULL;
    }
    
    Slice* resolved = malloc(sizeof(Slice) * t->ndims);
    for (usize d = 0; d < t->ndims; d++) {
        resolved[d] = slices[d];
        ResolveSlice(&resolved[d], t->dims[d]);
    }
    
    usize* new_dims = malloc(sizeof(usize) * t->ndims);
    for (usize d = 0; d < t->ndims; d++) {
        new_dims[d] = SliceLength(&resolved[d]);
    }
    
    Tensor* result = TensorNew(new_dims, t->ndims);
    usize* src_idx = calloc(t->ndims, sizeof(usize));
    usize* dst_idx = calloc(t->ndims, sizeof(usize));
    
    SliceCopyHelper(t, result, resolved, src_idx, dst_idx, 0);
    
    free(resolved);
    free(new_dims);
    free(src_idx);
    free(dst_idx);
    
    return result;
}

Tensor* TensorReshape(Tensor* t, usize* new_dims, usize new_ndims) {
    CHECK_NULL(t, "TensorReshape: cannot reshape NULL tensor\n", t);
    usize new_length = CalculateLength(new_dims, new_ndims);

    if (new_length != t->length) {
        fprintf(stderr, "TensorReshape: cannot reshape tensor of size %zu to size %zu \n", t->length, new_length);
        abort();
    }

    // Can only return a view if tensor is contiguous
    if (!TensorIsContiguous(t)) {
        // Not contiguous, need to make a copy
        Tensor* new_t = TensorNew(new_dims, new_ndims);
        // Copy data element by element
        usize* idx = calloc(t->ndims, sizeof(usize));
        for (usize i = 0; i < t->length; i++) {
            usize temp = i;
            for (isize d = t->ndims - 1; d >= 0; d--) {
                idx[d] = temp % t->dims[d];
                temp /= t->dims[d];
            }
            new_t->data[i] = TensorGet(t, idx);
        }
        free(idx);
        return new_t;
    }

    // Contiguous, can return a view with new shape
    Tensor* view = malloc(sizeof(Tensor));
    CHECK_NULL(view, "TensorReshape: failed to allocate view\n", NULL);

    view->ndims = new_ndims;
    view->dims = malloc(sizeof(usize) * new_ndims);
    CHECK_NULL(view->dims, "TensorReshape: failed to allocate dims\n", view);
    memcpy(view->dims, new_dims, sizeof(usize) * new_ndims);

    view->stride = malloc(sizeof(usize) * new_ndims);
    CHECK_NULL(view->stride, "TensorReshape: failed to allocate stride\n", view);
    InitStride(new_dims, new_ndims, view->stride);

    // Share data
    view->data = t->data;
    view->data_owner = NULL;
    view->offset = t->offset;
    view->length = t->length;

    // Share reference count
    view->ref_count = t->ref_count;
    if (view->ref_count != NULL) {
        (*view->ref_count)++;
    }

    return view;
}

Tensor* TensorTranspose(Tensor* t) {
    CHECK_NULL(t, "TensorTranspose: cannot transpose NULL tensor\n", NULL);

    if (t->ndims != 2) {
        fprintf(stderr, "TensorTranspose: expected 2D tensor, got %zuD\n", t->ndims);
        abort();
    }

    // Create a view with transposed dimensions and strides
    Tensor* view = malloc(sizeof(Tensor));
    CHECK_NULL(view, "TensorTranspose: failed to allocate view\n", NULL);

    view->ndims = t->ndims;
    view->dims = malloc(sizeof(usize) * view->ndims);
    CHECK_NULL(view->dims, "TensorTranspose: failed to allocate dims\n", view);

    view->stride = malloc(sizeof(usize) * view->ndims);
    CHECK_NULL(view->stride, "TensorTranspose: failed to allocate stride\n", view);

    // Swap dimensions and strides
    view->dims[0] = t->dims[1];
    view->dims[1] = t->dims[0];
    view->stride[0] = t->stride[1];
    view->stride[1] = t->stride[0];

    // Share data
    view->data = t->data;
    view->data_owner = NULL;  // This is a view
    view->offset = t->offset;
    view->length = t->length;

    // Share reference count
    view->ref_count = t->ref_count;
    if (view->ref_count != NULL) {
        (*view->ref_count)++;
    }

    return view;
}

Tensor* TensorSqueeze(Tensor* t) {
    CHECK_NULL(t, "TensorSqueeze: cannot squeeze NULL tensor\n", NULL);

    // Count non-1 dimensions
    usize new_ndims = 0;
    for (usize d = 0; d < t->ndims; d++) {
        if (t->dims[d] != 1) {
            new_ndims++;
        }
    }

    if (new_ndims == 0) {
        new_ndims = 1;
    }

    // Create a view with squeezed dimensions
    Tensor* view = malloc(sizeof(Tensor));
    CHECK_NULL(view, "TensorSqueeze: failed to allocate view\n", NULL);

    view->ndims = new_ndims;
    view->dims = malloc(sizeof(usize) * new_ndims);
    CHECK_NULL(view->dims, "TensorSqueeze: failed to allocate dims\n", view);

    view->stride = malloc(sizeof(usize) * new_ndims);
    CHECK_NULL(view->stride, "TensorSqueeze: failed to allocate stride\n", view);

    // Copy non-1 dimensions and their strides
    usize j = 0;
    for (usize d = 0; d < t->ndims; d++) {
        if (t->dims[d] != 1) {
            view->dims[j] = t->dims[d];
            view->stride[j] = t->stride[d];
            j++;
        }
    }

    // Handle scalar case
    if (j == 0) {
        view->dims[0] = 1;
        view->stride[0] = 1;
    }

    // Share data
    view->data = t->data;
    view->data_owner = NULL;
    view->offset = t->offset;
    view->length = t->length;

    // Share reference count
    view->ref_count = t->ref_count;
    if (view->ref_count != NULL) {
        (*view->ref_count)++;
    }

    return view;
}

Tensor* TensorUnsqueeze(Tensor* t, usize dim) {
    CHECK_NULL(t, "TensorUnsqueeze: cannot unsqueeze NULL tensor\n", NULL);

    if (dim > t->ndims) {
        fprintf(stderr, "TensorUnsqueeze: dim %zu out of range for %zuD tensor\n",
                dim, t->ndims);
        abort();
    }

    usize new_ndims = t->ndims + 1;

    // Create a view with unsqueezed dimensions
    Tensor* view = malloc(sizeof(Tensor));
    CHECK_NULL(view, "TensorUnsqueeze: failed to allocate view\n", NULL);

    view->ndims = new_ndims;
    view->dims = malloc(sizeof(usize) * new_ndims);
    CHECK_NULL(view->dims, "TensorUnsqueeze: failed to allocate dims\n", view);

    view->stride = malloc(sizeof(usize) * new_ndims);
    CHECK_NULL(view->stride, "TensorUnsqueeze: failed to allocate stride\n", view);

    // Insert dimension of size 1 at position dim
    for (usize d = 0; d < dim; d++) {
        view->dims[d] = t->dims[d];
        view->stride[d] = t->stride[d];
    }
    view->dims[dim] = 1;
    view->stride[dim] = (dim < t->ndims) ? t->stride[dim] : 1;
    for (usize d = dim; d < t->ndims; d++) {
        view->dims[d + 1] = t->dims[d];
        view->stride[d + 1] = t->stride[d];
    }

    // Share data
    view->data = t->data;
    view->data_owner = NULL;
    view->offset = t->offset;
    view->length = t->length;

    // Share reference count
    view->ref_count = t->ref_count;
    if (view->ref_count != NULL) {
        (*view->ref_count)++;
    }

    return view;
}

Tensor* TensorPermute(Tensor* t, usize* perm) {
    CHECK_NULL(t, "TensorPermute: cannot permute NULL tensor\n", NULL);
    CHECK_NULL(perm, "TensorPermute: perm is NULL\n", NULL);

    // Validate permutation
    bool* seen = calloc(t->ndims, sizeof(bool));
    for (usize d = 0; d < t->ndims; d++) {
        if (perm[d] >= t->ndims) {
            fprintf(stderr, "TensorPermute: invalid perm index %zu\n", perm[d]);
            free(seen);
            abort();
        }
        if (seen[perm[d]]) {
            fprintf(stderr, "TensorPermute: duplicate in perm\n");
            free(seen);
            abort();
        }
        seen[perm[d]] = true;
    }
    free(seen);

    // Create a view with permuted dimensions and strides
    Tensor* view = malloc(sizeof(Tensor));
    CHECK_NULL(view, "TensorPermute: failed to allocate view\n", NULL);

    view->ndims = t->ndims;
    view->dims = malloc(sizeof(usize) * view->ndims);
    CHECK_NULL(view->dims, "TensorPermute: failed to allocate dims\n", view);

    view->stride = malloc(sizeof(usize) * view->ndims);
    CHECK_NULL(view->stride, "TensorPermute: failed to allocate stride\n", view);

    // Permute dimensions and strides according to perm
    for (usize d = 0; d < t->ndims; d++) {
        view->dims[d] = t->dims[perm[d]];
        view->stride[d] = t->stride[perm[d]];
    }

    // Share data
    view->data = t->data;
    view->data_owner = NULL;
    view->offset = t->offset;
    view->length = t->length;

    // Share reference count
    view->ref_count = t->ref_count;
    if (view->ref_count != NULL) {
        (*view->ref_count)++;
    }

    return view;
}

DEFINE_REDUCE_ALL_IMPL(TensorSum,  0.0f,       result += a->data[i], "omp parallel for reduction(+:result) if(size > OMP_THRESHOLD)")
DEFINE_REDUCE_ALL_IMPL(TensorProd, 1.0f,       result *= a->data[i], "omp parallel for reduction(*:result) if(size > OMP_THRESHOLD)")

float TensorMean(Tensor* a) {
    float sum = TensorSum(a);
    return sum / a->length;
}

float TensorVar(Tensor* a, bool unbiased) {
    CHECK_NULL(a, "TensorVar: a is NULL\n", NULL);
    float mean = TensorMean(a);
    usize size = a->length;

    float var = 0.0f;
    #pragma omp parallel for simd reduction(+:var) if(size > OMP_THRESHOLD)
    for (usize i = 0; i < size; i++) {
        float diff = a->data[i] - mean;
        var += diff * diff;
    }
    return unbiased ? var / (size - 1) : var / size;
}

Tensor* TensorVarDim(Tensor* a, usize dim, bool keepdim, bool unbiased) {
    CHECK_NULL(a, "TensorVarDim: a is NULL\n", NULL);
    
    if (dim >= a->ndims) {
        fprintf(stderr, "TensorVarDim: dim %zu out of range\n", dim);
        abort();
    }
    
    Tensor* mean = TensorMeanDim(a, dim, true);  // Always keepdim for broadcasting
    
    usize old_ndims = a->ndims;
    usize new_ndims = keepdim ? old_ndims : old_ndims - 1;
    usize* new_dims = malloc(sizeof(usize) * new_ndims);
    
    usize j = 0;
    for (usize d = 0; d < old_ndims; d++) {
        if (d == dim) {
            if (keepdim) new_dims[j++] = 1;
        } else {
            new_dims[j++] = a->dims[d];
        }
    }
    
    Tensor* result = TensorNew(new_dims, new_ndims);
    usize result_size = result->length;
    usize reduce_size = a->dims[dim];
    usize reduce_stride = a->stride[dim];
    
    usize* stride_map = malloc(sizeof(usize) * new_ndims);
    j = 0;
    for (usize d = 0; d < old_ndims; d++) {
        if (d != dim || keepdim) {
            stride_map[j++] = (d == dim) ? 0 : a->stride[d];
        }
    }
    
    float divisor = unbiased ? (float)(reduce_size - 1) : (float)reduce_size;
    
    #pragma omp parallel for if(result_size > OMP_THRESHOLD)
    for (usize i = 0; i < result_size; i++) {
        usize base_offset = 0;
        usize temp = i;
        for (isize d = new_ndims - 1; d >= 0; d--) {
            usize idx = temp % result->dims[d];
            temp /= result->dims[d];
            base_offset += idx * stride_map[d];
        }
        
        float m = mean->data[i];
        float acc = 0.0f;
        for (usize r = 0; r < reduce_size; r++) {
            float diff = a->data[base_offset + r * reduce_stride] - m;
            acc += diff * diff;
        }
        result->data[i] = acc / divisor;
    }
    
    free(new_dims);
    free(stride_map);
    TensorFree(mean);
    return result;
}

float TensorStd(Tensor* a, bool unbiased) {
    return sqrtf(TensorVar(a, unbiased));
}

Tensor* TensorStdDim(Tensor* a, usize dim, bool keepdim, bool unbiased) {
    Tensor* var = TensorVarDim(a, dim, keepdim, unbiased);
    TensorSqrt_(var);
    return var;
}

DEFINE_REDUCE_ALL_IMPL(TensorMax,  a->data[0], if (a->data[i] > result) result = a->data[i], "omp parallel for reduction(max:result) if(size > OMP_THRESHOLD)")
DEFINE_REDUCE_ALL_IMPL(TensorMin,  a->data[0], if (a->data[i] < result) result = a->data[i], "omp parallel for reduction(min:result) if(size > OMP_THRESHOLD)")

DEFINE_ARG_REDUCE_ALL(TensorArgMax, >)
DEFINE_ARG_REDUCE_ALL(TensorArgMin, <)

static void MatMulNaive(float* a, float* b, float* c, usize M, usize K, usize N) {
    for (usize i = 0; i < M * N; i++) {
        c[i] = 0.0f;
    }
    
    #pragma omp parallel for if(M * N > OMP_THRESHOLD)
    for (usize i = 0; i < M; i++) {
        for (usize k = 0; k < K; k++) {
            float aik = a[i * K + k];
            #pragma omp simd
            for (usize j = 0; j < N; j++) {
                c[i * N + j] += aik * b[k * N + j];
            }
        }
    }
}

Tensor* TensorMatMul(Tensor* a, Tensor* b) {
    CHECK_NULL(a, "TensorMatMul: a is NULL\n", NULL);
    CHECK_NULL(b, "TensorMatMul: b is NULL\n", NULL);
    
    if (a->ndims < 2 || b->ndims < 2) {
        fprintf(stderr, "TensorMatMul: need at least 2D tensors\n");
        abort();
    }
    
    usize M = a->dims[a->ndims - 2];
    usize K1 = a->dims[a->ndims - 1];
    usize K2 = b->dims[b->ndims - 2];
    usize N = b->dims[b->ndims - 1];
    
    if (K1 != K2) {
        fprintf(stderr, "TensorMatMul: inner dimensions don't match (%zu vs %zu)\n", K1, K2);
        abort();
    }
    
    if (a->ndims != b->ndims) {
        fprintf(stderr, "TensorMatMul: batch dimensions must match\n");
        abort();
    }
    
    for (usize d = 0; d < a->ndims - 2; d++) {
        if (a->dims[d] != b->dims[d]) {
            fprintf(stderr, "TensorMatMul: batch dimension %zu doesn't match\n", d);
            abort();
        }
    }
    
    usize* result_dims = malloc(sizeof(usize) * a->ndims);
    for (usize d = 0; d < a->ndims - 2; d++) {
        result_dims[d] = a->dims[d];
    }
    result_dims[a->ndims - 2] = M;
    result_dims[a->ndims - 1] = N;
    
    Tensor* result = TensorNew(result_dims, a->ndims);
    
    usize batch_size = 1;
    for (usize d = 0; d < a->ndims - 2; d++) {
        batch_size *= a->dims[d];
    }
    
    usize a_matrix_size = M * K1;
    usize b_matrix_size = K1 * N;
    usize c_matrix_size = M * N;
    
    #pragma omp parallel for if(batch_size > 1)
    for (usize batch = 0; batch < batch_size; batch++) {
        float* a_ptr = a->data + batch * a_matrix_size;
        float* b_ptr = b->data + batch * b_matrix_size;
        float* c_ptr = result->data + batch * c_matrix_size;
        
        #ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K1,
                        1.0f, a_ptr, K1,
                        b_ptr, N,
                        0.0f, c_ptr, N);
        #else
            MatMulNaive(a_ptr, b_ptr, c_ptr, M, K1, N);
        #endif
    }
    
    free(result_dims);
    return result;
}

float TensorDot(Tensor* a, Tensor* b) {
    CHECK_NULL(a, "TensorDot: a is NULL\n", NULL);
    CHECK_NULL(b, "TensorDot: b is NULL\n", NULL);

    if (a->ndims != 1 || b->ndims != 1) {
        fprintf(stderr, "TensorDot: expected 1D tensors, got %zuD and %zuD\n", a->ndims, b->ndims);
        abort();
    }

    if (a->length != b->length) {
        fprintf(stderr, "TensorDot: length mismatch (%zu vs %zu)\n", a->length, b->length);
        abort();
    }

    float total = 0.0f;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data;
    usize size = a->length; 

    #pragma omp parallel for simd reduction(+:total) if(size > OMP_THRESHOLD)
    for (usize i = 0; i < size; i++) {
        total += a_data[i] * b_data[i];
    }

    return total;
}

Tensor* TensorSoftmax(Tensor* a, usize dim) {
    CHECK_NULL(a, "TensorSoftmax: a is NULL\n", NULL);
    
    if (dim >= a->ndims) {
        fprintf(stderr, "TensorSoftmax: dim %zu out of range for %zuD tensor\n", dim, a->ndims);
        abort();
    }
    
    Tensor* max_vals = TensorMaxDim(a, dim, true);
    Tensor* result = TensorCopy(a);
    
    usize reduce_size = a->dims[dim];
    usize reduce_stride = a->stride[dim];
    usize outer_size = a->length / reduce_size;
    
    // Compute stride to iterate over all positions except `dim`
    usize* stride_map = malloc(sizeof(usize) * max_vals->ndims);
    usize j = 0;
    for (usize d = 0; d < a->ndims; d++) {
        if (d != dim) {
            stride_map[j++] = a->stride[d];
        }
    }
    stride_map[j] = 0; // for keepdim=true case
    
    #pragma omp parallel for if(outer_size > OMP_THRESHOLD)
    for (usize i = 0; i < max_vals->length; i++) {
        usize base_offset = 0;
        usize temp = i;
        for (isize d = max_vals->ndims - 1; d >= 0; d--) {
            usize idx = temp % max_vals->dims[d];
            temp /= max_vals->dims[d];
            base_offset += idx * stride_map[d];
        }
        
        float max_val = max_vals->data[i];
        
        float sum = 0.0f;
        for (usize r = 0; r < reduce_size; r++) {
            usize offset = base_offset + r * reduce_stride;
            result->data[offset] = expf(a->data[offset] - max_val);
            sum += result->data[offset];
        }
        
        for (usize r = 0; r < reduce_size; r++) {
            usize offset = base_offset + r * reduce_stride;
            result->data[offset] /= sum;
        }
    }
    
    free(stride_map);
    TensorFree(max_vals);
    return result;
}

DEFINE_CLAMP_OP(TensorClamp, fminf(fmaxf(a_data[_i], min), max), fminf(fmaxf(a_val, min), max))

static void TensorPrintHelper(Tensor* t, usize* indices, usize depth, bool summarize, usize edge_items) {
    if (depth == t->ndims) {
        usize offset = 0;
        for (usize d = 0; d < t->ndims; d++) {
            offset += indices[d] * t->stride[d];
        }
        printf("%g", t->data[offset]);
        return;
    }
    
    printf("[");
    
    usize dim_size = t->dims[depth];
    bool skipMiddle = summarize && (dim_size > 2 * edge_items);
    
    for (usize i = 0; i < dim_size; i++) {
        if (skipMiddle && i == edge_items) {
            printf("...");
            i = dim_size - edge_items - 1;
            if (i < dim_size - 1) {
                printf(", ");
            }
            continue;
        }
        
        indices[depth] = i;
        TensorPrintHelper(t, indices, depth + 1, summarize, edge_items);
        
        if (i < dim_size - 1) {
            printf(", ");
        }
    }
    
    printf("]");
}

void TensorPrint(Tensor* t, usize threshold) {
    CHECK_NULL(t, "TensorPrint: tried to print NULL tensor\n", t);
    bool summarize = t->length > threshold;
    usize edge_items = 3;

    if (!t) {
        printf("NULL\n");
        return;
    }
    
    usize* indices = calloc(t->ndims, sizeof(usize));
    if (!indices) {
        fprintf(stderr, "TensorPrint: failed to allocate indices\n");
        return;
    }
    
    TensorPrintHelper(t, indices, 0, summarize, edge_items);
    printf("\n");
    
    free(indices);
}

// Broadcasting support

// Compute the broadcasted shape of two tensors following NumPy broadcasting rules
// Returns NULL if shapes are not compatible for broadcasting
// out_ndims is set to the number of dimensions in the result
usize* TensorBroadcastShape(Tensor* a, Tensor* b, usize* out_ndims) {
    CHECK_NULL(a, "TensorBroadcastShape: a is NULL\n", NULL);
    CHECK_NULL(b, "TensorBroadcastShape: b is NULL\n", NULL);
    CHECK_NULL(out_ndims, "TensorBroadcastShape: out_ndims is NULL\n", NULL);

    // Output has max number of dimensions
    *out_ndims = a->ndims > b->ndims ? a->ndims : b->ndims;
    usize* result = malloc(*out_ndims * sizeof(usize));
    CHECK_NULL(result, "TensorBroadcastShape: failed to allocate result\n", NULL);

    // Align dimensions from the right (trailing dimensions)
    // Process from right to left
    for (isize i = *out_ndims - 1; i >= 0; i--) {
        // Get dimension sizes, treating missing dimensions as 1
        isize a_idx = i - (*out_ndims - a->ndims);
        isize b_idx = i - (*out_ndims - b->ndims);

        usize a_dim = (a_idx >= 0) ? a->dims[a_idx] : 1;
        usize b_dim = (b_idx >= 0) ? b->dims[b_idx] : 1;

        // Broadcasting rules: dimensions must be equal or one of them must be 1
        if (a_dim == b_dim) {
            result[i] = a_dim;
        } else if (a_dim == 1) {
            result[i] = b_dim;
        } else if (b_dim == 1) {
            result[i] = a_dim;
        } else {
            // Incompatible dimensions
            fprintf(stderr, "TensorBroadcastShape: incompatible shapes at dimension %ld: %zu vs %zu\n",
                    (long)i, a_dim, b_dim);
            free(result);
            return NULL;
        }
    }

    return result;
}

// Create a view that broadcasts a tensor to a target shape using zero-stride trick
// Returns NULL if broadcast is not possible
Tensor* TensorBroadcastTo(Tensor* t, usize* target_shape, usize target_ndims) {
    CHECK_NULL(t, "TensorBroadcastTo: t is NULL\n", NULL);
    CHECK_NULL(target_shape, "TensorBroadcastTo: target_shape is NULL\n", NULL);
    CHECK_N_DIMS(target_ndims, "TensorBroadcastTo: target_ndims is 0\n");

    // Allocate the view
    Tensor* view = malloc(sizeof(Tensor));
    CHECK_NULL(view, "TensorBroadcastTo: failed to allocate view\n", NULL);

    view->ndims = target_ndims;
    view->dims = malloc(target_ndims * sizeof(usize));
    CHECK_NULL(view->dims, "TensorBroadcastTo: failed to allocate dims\n", view);

    view->stride = malloc(target_ndims * sizeof(usize));
    CHECK_NULL(view->stride, "TensorBroadcastTo: failed to allocate stride\n", view);

    // Copy target shape
    memcpy(view->dims, target_shape, target_ndims * sizeof(usize));

    // Compute strides with zero-stride trick for broadcasted dimensions
    // Align from the right
    usize length = 1;
    for (isize i = target_ndims - 1; i >= 0; i--) {
        isize t_idx = i - (target_ndims - t->ndims);

        if (t_idx < 0) {
            // This dimension doesn't exist in source - broadcast with stride 0
            view->stride[i] = 0;
        } else {
            usize t_dim = t->dims[t_idx];
            usize target_dim = target_shape[i];

            if (t_dim == target_dim) {
                // Dimensions match - use original stride
                view->stride[i] = t->stride[t_idx];
            } else if (t_dim == 1) {
                // Broadcasting from 1 to target_dim - use stride 0
                view->stride[i] = 0;
            } else {
                // Incompatible dimensions
                fprintf(stderr, "TensorBroadcastTo: cannot broadcast dimension %ld from %zu to %zu\n",
                        (long)i, t_dim, target_dim);
                free(view->dims);
                free(view->stride);
                free(view);
                return NULL;
            }
        }

        length *= target_shape[i];
    }

    view->length = length;

    // Share data with original tensor
    view->data = t->data;
    view->data_owner = NULL;  // This is a view
    view->offset = t->offset;

    // Share reference count
    view->ref_count = t->ref_count;
    if (view->ref_count != NULL) {
        (*view->ref_count)++;
    }

    return view;
}

void TensorFree(Tensor* t) {
    if (t == NULL) {
        return;
    }

    // Decrement reference count
    if (t->ref_count != NULL) {
        (*t->ref_count)--;

        // Only free data if this is the last reference and we own it
        if (*t->ref_count == 0) {
            if (t->data_owner != NULL) {
                free(t->data_owner);
            }
            free(t->ref_count);
        }
    } else {
        // Legacy path: no ref count, just free if we own the data
        if (t->data_owner != NULL) {
            free(t->data_owner);
        }
    }

    free(t->dims);
    free(t->stride);
    free(t);
}