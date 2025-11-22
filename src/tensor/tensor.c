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

#define TENSOR_LOOP(size, body) do { \
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

struct Tensor {
    float* data;
    usize* dims;
    usize ndims;
    usize length;
    usize* stride; 
};

#define CheckNull(ptr, msg, t) \
    if (!ptr) { \
        fprintf(stderr, msg); \
        if (t) { \
            TensorFree(t); \
        } \
        abort(); \
    } 

#define CheckNDims(ndims, msg) \
    if (ndims == 0) { \
        fprintf(stderr, msg); \
        abort(); \
    }

#define CheckEqualShape(a, b, fn_name) do { \
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
    CheckNull(dims, "TensorNew: dims is NULL\n", NULL);
    CheckNDims(ndims, "TensorNew: ndims is 0\n");
    usize length = CalculateLength(dims, ndims);

    Tensor* t = malloc(sizeof(Tensor));
    CheckNull(t, "TensorNew: failed to create Tensor struct\n", NULL);
    t->data = NULL; t->dims = NULL; t->stride = NULL; 
    
    t->dims = malloc(sizeof(usize) * ndims);
    CheckNull(t->dims, "TensorNew: failed to create Tensor dims array\n", t);
    memcpy(t->dims, dims, sizeof(usize) * ndims);
    t->ndims = ndims;

    t->data = calloc(length, sizeof(float));
    CheckNull(t->data, "TensorNew: failed to create data array\n", t);
    t->length = length;

    t->stride = malloc(sizeof(usize) * ndims);
    CheckNull(t->stride, "TensorNew: failed to create stride array\n", t);
    InitStride(dims, ndims, t->stride);

    return t;
}

Tensor* TensorFrom(usize* dims, usize ndims, float* data) {
    CheckNull(data, "TensorFrom: tried to create tensor from NULL data\n", NULL);
    Tensor* t = TensorNew(dims, ndims);

    memcpy(t->data, data, t->length * sizeof(float));

    return t;
}

Tensor* TensorCopy(Tensor* t) {
    CheckNull(t, "TensorCopy: tried to create null tensor\n", NULL);
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
    CheckNull(t, "TensorNDims: Tried to get dimension of NULL tensor\n", t);
    return t->ndims;
}

usize* TensorShape(Tensor *t) {
    CheckNull(t, "TensorShape: Tried to get shape of NULL tensor\n", t);
    return t->dims;
}

usize TensorSize(Tensor* t) {
    CheckNull(t, "TensorSize: Tried to get size of NULL tensor\n", t);
    return t->length;
}

usize TensorDim(Tensor* t, usize d) {
    CheckNull(t, "TensorDim: Tried to get dimension of NULL tensor\n", t);
    
    if (d >= t->ndims) {
        fprintf(stderr, "Tried to access out of bounds dimension %zu for tensor with %zu dimensions\n", d, t->ndims);
        TensorFree(t);
        abort();
    }

    return t->dims[d];
}

float* TensorData(Tensor* t) {
    CheckNull(t, "TensorData: NULL tensor\n", NULL);
    return t->data;
}

bool TensorShapeEqual(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorShapeEqual: Tried to get shape of NULL tensor\n", a);
    CheckNull(b, "TensorShapeEqual: Tried to get shape of NULL tensor\n", b);

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
    CheckNull(t, "TensorGet: Tried to index NULL Tensor", NULL);
    CheckNull(idx, "TensorGet: Tried to access Tensor ", t);

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
    CheckNull(t, "TensorGet: Tried to index NULL Tensor", NULL);
    CheckNull(idx, "TensorGet: Tried to access Tensor ", t);

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
    CheckNull(t, "TensorReshape: cannot reshape NULL tensor\n", t);
    usize new_length = CalculateLength(new_dims, new_ndims);
    
    if (new_length != t->length) {
        fprintf(stderr, "TensorReshape: cannot reshape tensor of size %zu to size %zu \n", t->length, new_length);
        abort();
    }

    Tensor* new_t = TensorNew(new_dims, new_ndims);
    memcpy(new_t->data, t->data, t->length * sizeof(float));
    return new_t;
}

Tensor* TensorTranspose(Tensor* t) {
    CheckNull(t, "TensorTranspose: cannot transpose NULL tensor\n", NULL);
    
    if (t->ndims != 2) {
        fprintf(stderr, "TensorTranspose: expected 2D tensor, got %zuD\n", t->ndims);
        abort();
    }
    
    usize new_dims[2] = {t->dims[1], t->dims[0]};
    Tensor* result = TensorNew(new_dims, 2);
    
    for (usize i = 0; i < t->dims[0]; i++) {
        for (usize j = 0; j < t->dims[1]; j++) {
            float val = TensorGet(t, SEQ(i, j));
            TensorSet(result, SEQ(j, i), val);
        }
    }
    
    return result;
}

Tensor* TensorSqueeze(Tensor* t) {
    CheckNull(t, "TensorSqueeze: cannot squeeze NULL tensor\n", NULL);
    
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
    
    usize* new_dims = malloc(sizeof(usize) * new_ndims);
    usize j = 0;

    for (usize d = 0; d < t->ndims; d++) {
        if (t->dims[d] != 1) {
            new_dims[j++] = t->dims[d];
        }
    }

    if (j == 0) {
        new_dims[0] = 1;
    }
    
    Tensor* result = TensorReshape(t, new_dims, new_ndims);
    free(new_dims);
    return result;
}

Tensor* TensorUnsqueeze(Tensor* t, usize dim) {
    CheckNull(t, "TensorUnsqueeze: cannot unsqueeze NULL tensor\n", NULL);
    
    if (dim > t->ndims) {
        fprintf(stderr, "TensorUnsqueeze: dim %zu out of range for %zuD tensor\n", 
                dim, t->ndims);
        abort();
    }
    
    usize new_ndims = t->ndims + 1;
    usize* new_dims = malloc(sizeof(usize) * new_ndims);
    
    for (usize d = 0; d < dim; d++) {
        new_dims[d] = t->dims[d];
    }
    new_dims[dim] = 1;
    for (usize d = dim; d < t->ndims; d++) {
        new_dims[d + 1] = t->dims[d];
    }
    
    Tensor* result = TensorReshape(t, new_dims, new_ndims);
    free(new_dims);
    return result;
}

Tensor* TensorPermute(Tensor* t, usize* perm) {
    CheckNull(t, "TensorPermute: cannot permute NULL tensor\n", NULL);
    CheckNull(perm, "TensorPermute: perm is NULL\n", NULL);
    
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
    
    usize* new_dims = malloc(sizeof(usize) * t->ndims);
    for (usize d = 0; d < t->ndims; d++) {
        new_dims[d] = t->dims[perm[d]];
    }
    
    Tensor* result = TensorNew(new_dims, t->ndims);
    
    usize* src_idx = malloc(sizeof(usize) * t->ndims);
    usize* dst_idx = malloc(sizeof(usize) * t->ndims);
    
    for (usize i = 0; i < t->length; i++) {
        usize temp = i;
        for (isize d = t->ndims - 1; d >= 0; d--) {
            src_idx[d] = temp % t->dims[d];
            temp /= t->dims[d];
        }
        
        for (usize d = 0; d < t->ndims; d++) {
            dst_idx[d] = src_idx[perm[d]];
        }
        
        TensorSet(result, dst_idx, TensorGet(t, src_idx));
    }
    
    free(new_dims);
    free(src_idx);
    free(dst_idx);
    return result;
}

Tensor* TensorAdd(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorAdd: a is NULL\n", NULL);
    CheckNull(b, "TensorAdd: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorAdd");

    Tensor* t = TensorZero(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data; 

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i] + b_data[_i]);
    return t;
}

void TensorAdd_(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorAdd_: a is NULL\n", NULL);
    CheckNull(b, "TensorAdd_: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorAdd_");

    float* restrict a_data = a->data;
    float* restrict b_data = b->data; 

    TENSOR_LOOP(a->length, a_data[_i] += b_data[_i]);
}

Tensor* TensorSub(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorSub: a is NULL\n", NULL);
    CheckNull(b, "TensorSub: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorSub");

    Tensor* t = TensorZero(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data; 

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i] - b_data[_i]);
    return t;
}

void TensorSub_(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorSub_: a is NULL\n", NULL);
    CheckNull(b, "TensorSub_: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorSub_");

    float* restrict a_data = a->data;
    float* restrict b_data = b->data; 

    TENSOR_LOOP(a->length, a_data[_i] -= b_data[_i]);
}

Tensor* TensorMul(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorMul: a is NULL\n", NULL);
    CheckNull(b, "TensorMul: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorMul");

    Tensor* t = TensorZero(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data; 

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i] * b_data[_i]);
    return t;
}

void TensorMul_(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorMul_: a is NULL\n", NULL);
    CheckNull(b, "TensorMul_: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorMul_");

    float* restrict a_data = a->data;
    float* restrict b_data = b->data; 

    TENSOR_LOOP(a->length, a_data[_i] *= b_data[_i]);
}

Tensor* TensorDiv(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorDiv: a is NULL\n", NULL);
    CheckNull(b, "TensorDiv: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorDiv");

    Tensor* t = TensorZero(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data; 

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i] / b_data[_i]);
    return t;
}

void TensorDiv_(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorDiv_: a is NULL\n", NULL);
    CheckNull(b, "TensorDiv_: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorDiv_");

    float* restrict a_data = a->data;
    float* restrict b_data = b->data; 

    TENSOR_LOOP(a->length, a_data[_i] /= b_data[_i]);
}

Tensor* TensorAddScalar(Tensor* a, float scalar) {
    CheckNull(a, "TensorAddScalar: a is NULL\n", NULL);
 
    Tensor* t = TensorZero(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i]  + scalar);
    return t;
}

void TensorAddScalar_(Tensor* a, float scalar) {
    CheckNull(a, "TensorAddScalar_: a is NULL\n", NULL);

    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, a_data[_i] += scalar);
}

Tensor* TensorSubScalar(Tensor* a, float scalar) {
    CheckNull(a, "TensorSubScalar: a is NULL\n", NULL);

    Tensor* t = TensorZero(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i] - scalar);
    return t;
}

void TensorSubScalar_(Tensor* a, float scalar) {
    CheckNull(a, "TensorSubScalar_: a is NULL\n", NULL);

    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, a_data[_i] -= scalar);
}

Tensor* TensorMulScalar(Tensor* a, float scalar) {
    CheckNull(a, "TensorMulScalar: a is NULL\n", NULL);
    Tensor* t = TensorZero(a->dims, a->ndims);

    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i] * scalar);
    return t;
}

void TensorMulScalar_(Tensor* a, float scalar) {
    CheckNull(a, "TensorMulScalar_: a is NULL\n", NULL);

    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, a_data[_i] *= scalar);
}

Tensor* TensorDivScalar(Tensor* a, float scalar) {
    CheckNull(a, "TensorDivScalar: a is NULL\n", NULL);
    Tensor* t = TensorZero(a->dims, a->ndims);

    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i] / scalar);
    return t;
}

void TensorDivScalar_(Tensor* a, float scalar) {
    CheckNull(a, "TensorDivScalar_: a is NULL\n", NULL);

    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, a_data[_i] /= scalar);
}

Tensor* TensorNeg(Tensor* a) {
    CheckNull(a, "TensorNeg: a is NULL\n", NULL);
    Tensor* t = TensorZero(a->dims, a->ndims);

    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i] * -1.0);
    return t;
}

void TensorNeg_(Tensor* a) {
    CheckNull(a, "TensorNeg_: a is NULL\n", NULL);

    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, a_data[_i] *= -1.0);
}

Tensor* TensorAbs(Tensor* a) {
    CheckNull(a, "TensorAbs: a is NULL\n", NULL);
    Tensor* t = TensorZero(a->dims, a->ndims);

    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = a_data[_i] * ((a_data[_i]>0.0) - (a_data[_i]<0.0)));
    return t;
}

void TensorAbs_(Tensor* a) {
    CheckNull(a, "TensorAbs_: a is NULL\n", NULL);

    float* restrict a_data = a->data;
    TENSOR_LOOP(a->length, a_data[_i] *= ((a_data[_i]>0.0) - (a_data[_i]<0.0)));
}

Tensor* TensorExp(Tensor* a) {
    CheckNull(a, "TensorExp: a is NULL\n", NULL);
    Tensor* t = TensorZero(a->dims, a->ndims);

    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = expf(a_data[_i]));

    return t;
}

void TensorExp_(Tensor* a) {
    CheckNull(a, "TensorExp_: a is NULL\n", NULL);

    float* restrict a_data = a->data;
    TENSOR_LOOP(a->length, a_data[_i] = expf(a_data[_i]));
}

Tensor* TensorLog(Tensor* a) {
    CheckNull(a, "TensorLog: a is NULL\n", NULL);
    Tensor* t = TensorZero(a->dims, a->ndims);

    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = logf(a_data[_i]));

    return t;
}

void TensorLog_(Tensor* a) {
    CheckNull(a, "TensorLog_: a is NULL\n", NULL);

    float* restrict a_data = a->data;
    TENSOR_LOOP(a->length, a_data[_i] = logf(a_data[_i]));
}

Tensor* TensorSqrt(Tensor* a) {
    CheckNull(a, "TensorSqrt: a is NULL\n", NULL);
    Tensor* t = TensorZero(a->dims, a->ndims);

    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = sqrtf(a_data[_i]));

    return t;
}

void TensorSqrt_(Tensor* a) {
    CheckNull(a, "TensorSqrt_: a is NULL\n", NULL);

    float* restrict a_data = a->data;
    TENSOR_LOOP(a->length, a_data[_i] = sqrtf(a_data[_i]));
}

Tensor* TensorPow(Tensor* a, float p) {
    CheckNull(a, "TensorSqrt: a is NULL\n", NULL);
    Tensor* t = TensorZero(a->dims, a->ndims);


    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = powf(a_data[_i], p));

    return t;
}

void TensorPow_(Tensor* a, float p) {
    CheckNull(a, "TensorPow_: a is NULL\n", NULL);

    float* restrict a_data = a->data;
    TENSOR_LOOP(a->length, a_data[_i] = powf(a_data[_i], p));
}

float TensorSum(Tensor* a) {
    CheckNull(a, "TensorSum: a is NULL\n", NULL);
    float total = 0.0f;
    usize size = a->length;
    
    #pragma omp parallel for simd reduction(+:total) if(size > OMP_THRESHOLD)
    for (usize i = 0; i < size; i++) {
        total += a->data[i];
    }
    return total;
}

Tensor* TensorSumDim(Tensor* a, usize dim, bool keepdim) {
    CheckNull(a, "TensorSumDim: a is NULL\n", NULL);
    
    if (dim >= a->ndims) {
        fprintf(stderr, "TensorSumDim: dim %zu out of range\n", dim);
        abort();
    }
    
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
            if (d == dim) {
                stride_map[j++] = 0;
            } else {
                stride_map[j++] = a->stride[d];
            }
        }
    }
    
    #pragma omp parallel for if(result_size > OMP_THRESHOLD)
    for (usize i = 0; i < result_size; i++) {
        usize base_offset = 0;
        usize temp = i;
        for (isize d = new_ndims - 1; d >= 0; d--) {
            usize idx = temp % result->dims[d];
            temp /= result->dims[d];
            base_offset += idx * stride_map[d];
        }
        
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (usize r = 0; r < reduce_size; r++) {
            sum += a->data[base_offset + r * reduce_stride];
        }
        
        result->data[i] = sum;
    }
    
    free(new_dims);
    free(stride_map);
    return result;
}

float TensorMean(Tensor* a) {
    float sum = TensorSum(a);
    return sum / a->length;
}

Tensor* TensorMeanDim(Tensor* a, usize dim, bool keepdim) {
    CheckNull(a, "TensorMeanDim: a is NULL\n", NULL);
    
    if (dim >= a->ndims) {
        fprintf(stderr, "TensorMeanDim: dim %zu out of range\n", dim);
        abort();
    }
    
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
    
    #pragma omp parallel for if(result_size > OMP_THRESHOLD)
    for (usize i = 0; i < result_size; i++) {
        usize base_offset = 0;
        usize temp = i;
        for (isize d = new_ndims - 1; d >= 0; d--) {
            usize idx = temp % result->dims[d];
            temp /= result->dims[d];
            base_offset += idx * stride_map[d];
        }
        
        float sum = 0.0f;
        for (usize r = 0; r < reduce_size; r++) {
            sum += a->data[base_offset + r * reduce_stride];
        }
        
        result->data[i] = sum / reduce_size;
    }
    
    free(new_dims);
    free(stride_map);
    return result;
}


float TensorMax(Tensor* a) {
    CheckNull(a, "TensorMax: a is NULL\n", NULL);
    usize size = a->length;
    
    float max_val = a->data[0];

    #pragma omp parallel for reduction(max:max_val) if(size > OMP_THRESHOLD)
    for (usize i = 1; i < size; i++) {
        if (a->data[i] > max_val) {
            max_val = a->data[i];
        }
    }

    return max_val;
}


Tensor* TensorMaxDim(Tensor* a, usize dim, bool keepdim) {
    CheckNull(a, "TensorMaxDim: a is NULL\n", NULL);
    
    if (dim >= a->ndims) {
        fprintf(stderr, "TensorMaxDim: dim %zu out of range\n", dim);
        abort();
    }
    
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
    
    #pragma omp parallel for if(result_size > OMP_THRESHOLD)
    for (usize i = 0; i < result_size; i++) {
        usize base_offset = 0;
        usize temp = i;
        for (isize d = new_ndims - 1; d >= 0; d--) {
            usize idx = temp % result->dims[d];
            temp /= result->dims[d];
            base_offset += idx * stride_map[d];
        }
        
        float max_val = a->data[base_offset];
        for (usize r = 1; r < reduce_size; r++) {
            float val = a->data[base_offset + r * reduce_stride];
            if (val > max_val) max_val = val;
        }
        
        result->data[i] = max_val;
    }
    
    free(new_dims);
    free(stride_map);
    return result;
}

float TensorMin(Tensor* a) {
        CheckNull(a, "TensorMin: a is NULL\n", NULL);
    usize size = a->length;
    
    float min_val = a->data[0];

    #pragma omp parallel for reduction(min:min_val) if(size > OMP_THRESHOLD)
    for (usize i = 1; i < size; i++) {
        if (a->data[i] < min_val) {
            min_val = a->data[i];
        }
    }

    return min_val;
}

Tensor* TensorMinDim(Tensor* a, usize dim, bool keepdim) {
    CheckNull(a, "TensorMinDim: a is NULL\n", NULL);
    
    if (dim >= a->ndims) {
        fprintf(stderr, "TensorMinDim: dim %zu out of range\n", dim);
        abort();
    }
    
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
    
    #pragma omp parallel for if(result_size > OMP_THRESHOLD)
    for (usize i = 0; i < result_size; i++) {
        usize base_offset = 0;
        usize temp = i;
        for (isize d = new_ndims - 1; d >= 0; d--) {
            usize idx = temp % result->dims[d];
            temp /= result->dims[d];
            base_offset += idx * stride_map[d];
        }
        
        float min_val = a->data[base_offset];
        for (usize r = 1; r < reduce_size; r++) {
            float val = a->data[base_offset + r * reduce_stride];
            if (val < min_val) min_val = val;
        }
        
        result->data[i] = min_val;
    }
    
    free(new_dims);
    free(stride_map);
    return result;
}

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
    CheckNull(a, "TensorMatMul: a is NULL\n", NULL);
    CheckNull(b, "TensorMatMul: b is NULL\n", NULL);
    
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
    CheckNull(a, "TensorDot: a is NULL\n", NULL);
    CheckNull(b, "TensorDot: b is NULL\n", NULL);

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

Tensor* TensorEq(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorEq: a is NULL\n", NULL);
    CheckNull(b, "TensorEq: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorEq");

    Tensor* t = TensorNew(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data;

    TENSOR_LOOP(a->length, t_data[_i] = (a_data[_i] == b_data[_i]) ? 1.0f : 0.0f);
    return t;
}

Tensor* TensorGt(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorGt: a is NULL\n", NULL);
    CheckNull(b, "TensorGt: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorGt");

    Tensor* t = TensorNew(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data;

    TENSOR_LOOP(a->length, t_data[_i] = (a_data[_i] > b_data[_i]) ? 1.0f : 0.0f);
    return t;
}

Tensor* TensorLt(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorLt: a is NULL\n", NULL);
    CheckNull(b, "TensorLt: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorLt");

    Tensor* t = TensorNew(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data;

    TENSOR_LOOP(a->length, t_data[_i] = (a_data[_i] < b_data[_i]) ? 1.0f : 0.0f);
    return t;
}

Tensor* TensorGe(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorGe: a is NULL\n", NULL);
    CheckNull(b, "TensorGe: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorGe");

    Tensor* t = TensorNew(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data;

    TENSOR_LOOP(a->length, t_data[_i] = (a_data[_i] >= b_data[_i]) ? 1.0f : 0.0f);
    return t;
}

Tensor* TensorLe(Tensor* a, Tensor* b) {
    CheckNull(a, "TensorLe: a is NULL\n", NULL);
    CheckNull(b, "TensorLe: b is NULL\n", NULL);
    CheckEqualShape(a, b, "TensorLe");

    Tensor* t = TensorNew(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    float* restrict b_data = b->data;

    TENSOR_LOOP(a->length, t_data[_i] = (a_data[_i] <= b_data[_i]) ? 1.0f : 0.0f);
    return t;
}

Tensor* TensorRelu(Tensor* a) {
    CheckNull(a, "TensorRelu: a is NULL\n", NULL);
    Tensor* t = TensorNew(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = fmaxf(0.0, a_data[_i]));
    return t;
}

void TensorRelu_(Tensor* a) {
    CheckNull(a, "TensorRelu_: a is NULL\n", NULL);
    float* restrict a_data = a->data;
    TENSOR_LOOP(a->length, a_data[_i] = fmaxf(0.0, a_data[_i]));
}

Tensor* TensorSigmoid(Tensor* a) {
    CheckNull(a, "TensorSigmoid: a is NULL\n", NULL);
    Tensor* t = TensorNew(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = 1 / (1 + expf(-a_data[_i])));
    return t;
}

void TensorSigmoid_(Tensor* a) {
    CheckNull(a, "TensorSigmoid_: a is NULL\n", NULL);
    float* restrict a_data = a->data;
    TENSOR_LOOP(a->length, a_data[_i] = 1 / (1 + expf(-a_data[_i])));
}

Tensor* TensorTanh(Tensor* a) {
    CheckNull(a, "TensorTanh: a is NULL\n", NULL);
    Tensor* t = TensorNew(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;

    TENSOR_LOOP(a->length, t_data[_i] = tanhf(a_data[_i]));
    return t;
}

void TensorTanh_(Tensor* a) {
    CheckNull(a, "TensorTanh_: a is NULL\n", NULL);
    float* restrict a_data = a->data;
    TENSOR_LOOP(a->length, a_data[_i] = tanhf(a_data[_i]));
}

Tensor* TensorSoftmax(Tensor* a, usize dim) {
        CheckNull(a, "TensorSoftmax: a is NULL\n", NULL);
    
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

Tensor* TensorClamp(Tensor* a, float min, float max) {
    CheckNull(a, "TensorClamp: a is NULL\n", NULL);
    
    Tensor* t = TensorNew(a->dims, a->ndims);
    float* restrict t_data = t->data;
    float* restrict a_data = a->data;
    
    TENSOR_LOOP(a->length, t_data[_i] = fminf(fmaxf(a_data[_i], min), max));
    
    return t;
}

void TensorClamp_(Tensor* a, float min, float max) {
    CheckNull(a, "TensorClamp_: a is NULL\n", NULL);
    
    float* restrict a_data = a->data;
    
    TENSOR_LOOP(a->length, a_data[_i] = fminf(fmaxf(a_data[_i], min), max));
}

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
    CheckNull(t, "TensorPrint: tried to print NULL tensor\n", t);
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

void TensorFree(Tensor* t) {
    if (t == NULL) { 
        return; 
    }

    free(t->data);
    free(t->dims);
    free(t->stride);
    free(t);
}