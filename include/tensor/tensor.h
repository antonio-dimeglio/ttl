#ifndef TTL_TENSOR_H
#define TTL_TENSOR_H
#include "core/types.h"
#include "slice.h"
#include <omp.h>
#include <math.h>

typedef struct Tensor Tensor; 

/*
*   Macro to define indexes to TensorGet and TensorSet or
*   equivalently a shape for TensorNew
*   by writing SEQ(idx1, idx2, ...)
*/
#define SEQ(...) (usize[]){__VA_ARGS__}

/*
*   Macro that can be used to _replace_ (free after use) of
*   tensor that are not needed after certain operations.
*   Example:
*   Tensor* t = TensorNew(SEQ(3, 4), 2);
*   TensorReplace(t, TensorTranspose(t));  // t is now transposed, old freed
*   TensorReplace(t, TensorSqueeze(t));    // chain operations   
*/
#define TensorReplace(ptr, expr) do { \
    Tensor* _new = (expr);            \
    TensorFree(ptr);                  \
    (ptr) = _new;                     \
} while(0)


/*
*   Entry point for the creation of a new tensor.
*   This method initializes the underlying data to zero everywhere.
*/
Tensor* TensorNew(usize* dims, usize ndims);

/*
*   Create a new tensor with the given structure.
*   Note: all operations expect tensors to be in row major order.
*/
Tensor* TensorFrom(usize* dims, usize ndims, float* data);

/*
*   Returns a deep copy of a tensor.
*/
Tensor* TensorCopy(Tensor* t);

/*
*   Returns a tensor with all values initialized to zero.
*/
Tensor* TensorZero(usize* dims, usize ndims);

/*
*   Returns a tensor with all values initialized to one.
*/
Tensor* TensorOnes(usize* dims, usize ndims);

/*
*   Returns a tensor with all values initialized to _value_.
*/
Tensor* TensorFill(usize* dims, usize ndims, float value);

/*
*   Returns the number of dimensions of a tensor.
*/
usize TensorNDims(Tensor* t);

/*
*   Returns the shape array of a tensor.
*/
usize* TensorShape(Tensor *t);

/*
*   Returns the total number of elements of a tensor.
*/
usize TensorSize(Tensor* t);

/*
*   Returns the number of elements along a given dimension of a tensor.
*/
usize TensorDim(Tensor* t, usize d);

/*
*   Returns a pointer to the underlying data array.
*/
float* TensorData(Tensor* t);

/*
*   Returns true if a and b shape match, false otherwise.
*/
bool TensorShapeEqual(Tensor* a, Tensor* b);

/*
*   Returns the element at the given index (aborts if illegal index is accessed).
*   The index array is expected to be of the same size of the tensor ndims.
*/
float TensorGet(Tensor* t, usize* idx);

/*
*   Sets the tensor at the given index to the passed valeu (aborts if illegal index is accessed).
*   The index array is expected to be of the same size of the tensor ndims.
*/
void TensorSet(Tensor* t, usize* idx, float value);

/*
*   Returns a deep copy of a tensor slice. 
*   Example:
*       Tensor* s = TensorSlice(t, SLICES(RANGE(1, 3), ALL, RANGE(2, 5)));
*
*/
Tensor* TensorSlice(Tensor* t, Slice* slices, usize nslices);

/*
*   Returns a new tensor with the same data but different shape
*   The size of both tensor must match.
*/
Tensor* TensorReshape(Tensor* t, usize* new_dims, usize new_ndims);

/*
*   Performs transposition for 2D tensor.
*   Returns a new tensor transposed w.r.t. t.
*/
Tensor* TensorTranspose(Tensor* t);

/* 
*   Returns a new tensors without dimensions of size 1
*   Example: [1, 3, 1, 4] -> [3, 4]
*/
Tensor* TensorSqueeze(Tensor* t);

/* 
*   Returns a new tensors with a new dimension of size 1 at position dim
*   Example: [3, 4] with dim=0 -> [1, 3, 4]
*/
Tensor* TensorUnsqueeze(Tensor* t, usize dim);

/* 
*   General permute - reorder dimensions
*   e.g., for [2,3,4] tensor, perm=[2,0,1] gives [4,2,3]
*/ 
Tensor* TensorPermute(Tensor* t, usize* perm);

/*
*   Returns a new tensor: result = a + b (elementwise)
*   Tensors must have the same shape.
*/
Tensor* TensorAdd(Tensor* a, Tensor* b);

/*
*   In-place addition: a = a + b (elementwise)
*   Tensors must have the same shape.
*/
void TensorAdd_(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result = a - b (elementwise)
*   Tensors must have the same shape.
*/
Tensor* TensorSub(Tensor* a, Tensor* b);

/*
*   In-place subtraction: a = a - b (elementwise)
*   Tensors must have the same shape.
*/
void TensorSub_(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result = a * b (elementwise, Hadamard product)
*   Tensors must have the same shape.
*/
Tensor* TensorMul(Tensor* a, Tensor* b);

/*
*   In-place multiplication: a = a * b (elementwise)
*   Tensors must have the same shape.
*/
void TensorMul_(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result = a / b (elementwise)
*   Tensors must have the same shape.
*/
Tensor* TensorDiv(Tensor* a, Tensor* b);

/*
*   In-place division: a = a / b (elementwise)
*   Tensors must have the same shape.
*/
void TensorDiv_(Tensor* a, Tensor* b);

// ============ Scalar Arithmetic ============

/*
*   Returns a new tensor: result = a + scalar (elementwise)
*/
Tensor* TensorAddScalar(Tensor* a, float scalar);

/*
*   In-place scalar addition: a = a + scalar
*/
void TensorAddScalar_(Tensor* a, float scalar);

/*
*   Returns a new tensor: result = a - scalar (elementwise)
*/
Tensor* TensorSubScalar(Tensor* a, float scalar);

/*
*   In-place scalar subtraction: a = a - scalar
*/
void TensorSubScalar_(Tensor* a, float scalar);

/*
*   Returns a new tensor: result = a * scalar (elementwise)
*/
Tensor* TensorMulScalar(Tensor* a, float scalar);

/*
*   In-place scalar multiplication: a = a * scalar
*/
void TensorMulScalar_(Tensor* a, float scalar);

/*
*   Returns a new tensor: result = a / scalar (elementwise)
*/
Tensor* TensorDivScalar(Tensor* a, float scalar);

/*
*   In-place scalar division: a = a / scalar
*/
void TensorDivScalar_(Tensor* a, float scalar);


/*
*   Returns a new tensor: result = -a (elementwise negation)
*/
Tensor* TensorNeg(Tensor* a);

/*
*   In-place negation: a = -a
*/
void TensorNeg_(Tensor* a);

/*
*   Returns a new tensor: result = |a| (elementwise absolute value)
*/
Tensor* TensorAbs(Tensor* a);

/*
*   In-place absolute value: a = |a|
*/
void TensorAbs_(Tensor* a);

/*
*   Returns a new tensor: result = exp(a) (elementwise)
*/
Tensor* TensorExp(Tensor* a);

/*
*   In-place exponential: a = exp(a)
*/
void TensorExp_(Tensor* a);

/*
*   Returns a new tensor: result = log(a) (elementwise natural log)
*/
Tensor* TensorLog(Tensor* a);

/*
*   In-place natural log: a = log(a)
*/
void TensorLog_(Tensor* a);

/*
*   Returns a new tensor: result = sqrt(a) (elementwise)
*/
Tensor* TensorSqrt(Tensor* a);

/*
*   In-place square root: a = sqrt(a)
*/
void TensorSqrt_(Tensor* a);

/*
*   Returns a new tensor: result = a^p (elementwise power)
*/
Tensor* TensorPow(Tensor* a, float p);

/*
*   In-place power: a = a^p
*/
void TensorPow_(Tensor* a, float p);

/*
*   Returns the sum of all elements.
*/
float TensorSum(Tensor* a);

/*
*   Returns a new tensor with sum along specified dimension.
*   If keepdim is true, the reduced dimension is kept as size 1.
*/
Tensor* TensorSumDim(Tensor* a, usize dim, bool keepdim);

/*
*   Returns the mean of all elements.
*/
float TensorMean(Tensor* a);

/*
*   Returns a new tensor with mean along specified dimension.
*/
Tensor* TensorMeanDim(Tensor* a, usize dim, bool keepdim);

/*
*   Returns the maximum value of all elements.
*/
float TensorMax(Tensor* a);

/*
*   Returns a new tensor with max along specified dimension.
*/
Tensor* TensorMaxDim(Tensor* a, usize dim, bool keepdim);

/*
*   Returns the minimum value of all elements.
*/
float TensorMin(Tensor* a);

/*
*   Returns a new tensor with min along specified dimension.
*/
Tensor* TensorMinDim(Tensor* a, usize dim, bool keepdim);


/*
*   Matrix multiplication: result = a @ b
*   a must be shape [..., M, K], b must be shape [..., K, N]
*   Returns tensor of shape [..., M, N]
*/
Tensor* TensorMatMul(Tensor* a, Tensor* b);

/*
*   Dot product of two 1D tensors.
*/
float TensorDot(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result[i] = (a[i] == b[i]) ? 1.0 : 0.0
*/
Tensor* TensorEq(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result[i] = (a[i] > b[i]) ? 1.0 : 0.0
*/
Tensor* TensorGt(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result[i] = (a[i] < b[i]) ? 1.0 : 0.0
*/
Tensor* TensorLt(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result[i] = (a[i] >= b[i]) ? 1.0 : 0.0
*/
Tensor* TensorGe(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result[i] = (a[i] <= b[i]) ? 1.0 : 0.0
*/
Tensor* TensorLe(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result = max(0, a) (ReLU)
*/
Tensor* TensorRelu(Tensor* a);

/*
*   In-place ReLU: a = max(0, a)
*/
void TensorRelu_(Tensor* a);

/*
*   Returns a new tensor: result = sigmoid(a) = 1 / (1 + exp(-a))
*/
Tensor* TensorSigmoid(Tensor* a);

/*
*   In-place sigmoid
*/
void TensorSigmoid_(Tensor* a);

/*
*   Returns a new tensor: result = tanh(a)
*/
Tensor* TensorTanh(Tensor* a);

/*
*   In-place tanh
*/
void TensorTanh_(Tensor* a);

/*
*   Returns a new tensor: softmax along specified dimension
*   result[i] = exp(a[i]) / sum(exp(a))
*/
Tensor* TensorSoftmax(Tensor* a, usize dim);

/*
*   Returns a new tensor with values clamped to [min, max]
*/
Tensor* TensorClamp(Tensor* a, float min, float max);

/*
*   In-place clamp: a[i] = clamp(a[i], min, max)
*/
void TensorClamp_(Tensor* a, float min, float max);

/*
* Prints the content of a tensor, if the size of array
* is over the threshold, only a summary is printed (numpy style).
*/
void TensorPrint(Tensor* t, usize threshold);
void TensorFree(Tensor* t);
#endif 