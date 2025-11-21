#include "tensor/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
    
    // New dims
    usize* new_dims = malloc(sizeof(usize) * t->ndims);
    for (usize d = 0; d < t->ndims; d++) {
        new_dims[d] = t->dims[perm[d]];
    }
    
    Tensor* result = TensorNew(new_dims, t->ndims);
    
    // Copy with permuted indices
    usize* src_idx = malloc(sizeof(usize) * t->ndims);
    usize* dst_idx = malloc(sizeof(usize) * t->ndims);
    
    for (usize i = 0; i < t->length; i++) {
        // Convert flat index to src indices
        usize temp = i;
        for (isize d = t->ndims - 1; d >= 0; d--) {
            src_idx[d] = temp % t->dims[d];
            temp /= t->dims[d];
        }
        
        // Permute indices
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