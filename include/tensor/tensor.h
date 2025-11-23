#ifndef TTL_TENSOR_H
#define TTL_TENSOR_H
#include "core/types.h"
#include "random/random.h"
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
*   Returns a tensor with all values initialized from values sampled from a normal distribution.
*/
Tensor* TensorRandN(usize* dims, usize ndims, float mean, float std);

/*
*   Returns a tensor with all values initialized from values sampled from a uniform distribution.
*/
Tensor* TensorRandU(usize* dims, usize ndims, float low, float high);

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
*   Returns the stride along a given dimension of a tensor.
*/
usize TensorStride(Tensor* t, usize d);

/*
*   Returns a pointer to the underlying data array.
*/
float* TensorData(Tensor* t);

/*
*   Returns true if the tensor is a view (doesn't own its data).
*/
bool TensorIsView(Tensor* t);

/*
*   Returns true if the tensor data is contiguous in memory.
*/
bool TensorIsContiguous(Tensor* t);

/*
*   Returns the base tensor if this is a view, or the tensor itself if it owns data.
*/
Tensor* TensorBase(Tensor* t);

/*
*   Creates a view of the tensor (shares data, increments reference count).
*/
Tensor* TensorView(Tensor* t);

/*
*   Returns a contiguous copy if tensor is not contiguous, otherwise returns a view.
*/
Tensor* TensorAsContiguous(Tensor* t);

/*
*   Broadcasts tensor to a target shape. Returns a view with modified strides.
*   Uses zero-stride trick for broadcasted dimensions (no data copying).
*   Follows NumPy broadcasting rules.
*/
Tensor* TensorBroadcastTo(Tensor* t, usize* target_shape, usize target_ndims);

/*
*   Checks if two tensors are broadcast-compatible and returns the broadcast shape.
*   Returns NULL if shapes are incompatible. Caller must free the returned array.
*/
usize* TensorBroadcastShape(Tensor* a, Tensor* b, usize* out_ndims);

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
*   Returns a new tensor: result = a^p (elementwise power, scalar exponent)
*/
Tensor* TensorPow(Tensor* a, float p);

/*
*   In-place power: a = a^p (scalar exponent)
*/
void TensorPow_(Tensor* a, float p);

/*
*   Returns a new tensor: result = a^b (elementwise power, tensor exponent)
*/
Tensor* TensorPowT(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result = sin(a) (elementwise)
*/
Tensor* TensorSin(Tensor* a);

/*
*   In-place sin: a = sin(a)
*/
void TensorSin_(Tensor* a);

/*
*   Returns a new tensor: result = asin(a) (elementwise)
*/
Tensor* TensorAsin(Tensor* a);

/*
*   In-place sin: a = asin(a)
*/
void TensorAsin_(Tensor* a);

/*
*   Returns a new tensor: result = acos(a) (elementwise)
*/
Tensor* TensorAcos(Tensor* a);

/*
*   In-place sin: a = acos(a)
*/
void TensorAcos_(Tensor* a);

/*
*   Returns a new tensor: result = atan(a) (elementwise)
*/
Tensor* TensorAtan(Tensor* a);

/*
*   In-place atan: a = atan(a)
*/
void TensorAtan_(Tensor* a);

/*
*   Returns a new tensor: result = atanh(a) (elementwise)
*/
Tensor* TensorAtanh(Tensor* a);

/*
*   In-place atanh: a = atanh(a)
*/
void TensorAtanh_(Tensor* a);

/*
*   Returns a new tensor: result = sinh(a) (elementwise)
*/
Tensor* TensorSinh(Tensor* a);

/*
*   In-place sin: a = sinh(a)
*/
void TensorSinh_(Tensor* a);

/*
*   Returns a new tensor: result = cosh(a) (elementwise)
*/
Tensor* TensorCosh(Tensor* a);

/*
*   In-place sin: a = cosh(a)
*/
void TensorCosh_(Tensor* a);

/*
*   Returns a new tensor: result = cos(a) (elementwise)
*/
Tensor* TensorCos(Tensor* a);

/*
*   In-place cos: a = cos(a)
*/
void TensorCos_(Tensor* a);

/*
*   Returns a new tensor: result = tan(a) (elementwise)
*/
Tensor* TensorTan(Tensor* a);

/*
*   In-place sin: a = tan(a)
*/
void TensorTan_(Tensor* a);

/*
*   Returns a new tensor: result = floor(a) (elementwise)
*/
Tensor* TensorFloor(Tensor* a);

/*
*   In-place floor: a = floor(a)
*/
void TensorFloor_(Tensor* a);

/*
*   Returns a new tensor: result = ceil(a) (elementwise)
*/
Tensor* TensorCeil(Tensor* a);

/*
*   In-place ceil: a = ceil(a)
*/
void TensorCeil_(Tensor* a);

/*
*   Returns a new tensor: result = round(a) (elementwise)
*/
Tensor* TensorRound(Tensor* a);

/*
*   In-place round: a = round(a)
*/
void TensorRound_(Tensor* a);

/*
*   Returns a new tensor: result = a % b (elementwise modulo)
*/
Tensor* TensorMod(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result = atan2(a, b) (elementwise)
*/
Tensor* TensorAtan2(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result = min(a, b) (elementwise minimum)
*/
Tensor* TensorMin2(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result = max(a, b) (elementwise maximum)
*/
Tensor* TensorMax2(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result = isnan(a) (elementwise)
*/
Tensor* TensorIsNan(Tensor* a);

/*
*   Returns a new tensor: result = isinf(a) (elementwise)
*/
Tensor* TensorIsInf(Tensor* a);

/*
*   Returns a new tensor: result = sign(a) (elementwise)
*/
Tensor* TensorSign(Tensor* a);

/*
*   In-place sign: a = sign(a)
*/
void TensorSign_(Tensor* a);

/*
*   Returns a new tensor: result = 1/a (elementwise)
*/
Tensor* TensorReciprocal(Tensor* a);

/*
*   In-place reciprocal: a = 1/a
*/
void TensorReciprocal_(Tensor* a);

/*
*   Returns a new tensor: result = a^2 (elementwise)
*/
Tensor* TensorSquare(Tensor* a);

/*
*   In-place square: a = a^2
*/
void TensorSquare_(Tensor* a);

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
*   Returns the product of all elements.
*/
float TensorProd(Tensor* a);

/*
*   Returns a new tensor with product along specified dimension.
*   If keepdim is true, the reduced dimension is kept as size 1.
*/
Tensor* TensorProdDim(Tensor* a, usize dim, bool keepdim);

/*
*   Returns the mean of all elements.
*/
float TensorMean(Tensor* a);

/*
*   Returns a new tensor with mean along specified dimension.
*/
Tensor* TensorMeanDim(Tensor* a, usize dim, bool keepdim);

/*
*   Returns the variance of all elements.
*   If unbiased is set to true, the variance is computed by dividing by N-1, n otherwise
*/
float TensorVar(Tensor* a, bool unbiased);

/*
*   Returns a new tensor with variance along specified dimension.
*/
Tensor* TensorVarDim(Tensor* a, usize dim, bool keepdim, bool unbiased);

/*
*   Returns the standard deviation of all elements.
*   If unbiased is set to true, the variance is computed by dividing by N-1, n otherwise
*/
float TensorStd(Tensor* a, bool unbiased);

/*
*   Returns a new tensor with variance along specified dimension.
*/
Tensor* TensorStdDim(Tensor* a, usize dim, bool keepdim, bool unbiased);

/*
*   Returns the maximum value of all elements.
*/
float TensorMax(Tensor* a);

/*
*   Returns a new tensor with max along specified dimension.
*/
Tensor* TensorMaxDim(Tensor* a, usize dim, bool keepdim);

/*
*   Returns the dimension of the max value in the tensor.
*/
float TensorArgMax(Tensor* a);

/*
*   Returns a new tensor with argmax along specified dimension.
*/
Tensor* TensorArgMaxDim(Tensor* a, usize dim, bool keepdim);


/*
*   Returns the minimum value of all elements.
*/
float TensorMin(Tensor* a);

/*
*   Returns a new tensor with min along specified dimension.
*/
Tensor* TensorMinDim(Tensor* a, usize dim, bool keepdim);

/*
*   Returns the dimension of the min value in the tensor.
*/
float TensorArgMin(Tensor* a);

/*
*   Returns a new tensor with argmin along specified dimension.
*/
Tensor* TensorArgMinDim(Tensor* a, usize dim, bool keepdim);

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
*   Returns a new tensor: result[i] = (a[i] != b[i]) ? 1.0 : 0.0
*/
Tensor* TensorNe(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result[i] = (a[i] == scalar) ? 1.0 : 0.0
*/
Tensor* TensorEqScalar(Tensor* a, float scalar);

/*
*   Returns a new tensor: result[i] = (a[i] != scalar) ? 1.0 : 0.0
*/
Tensor* TensorNeScalar(Tensor* a, float scalar);

/*
*   Returns a new tensor: result[i] = (a[i] > scalar) ? 1.0 : 0.0
*/
Tensor* TensorGtScalar(Tensor* a, float scalar);

/*
*   Returns a new tensor: result[i] = (a[i] < scalar) ? 1.0 : 0.0
*/
Tensor* TensorLtScalar(Tensor* a, float scalar);

/*
*   Returns a new tensor: result[i] = (a[i] >= scalar) ? 1.0 : 0.0
*/
Tensor* TensorGeScalar(Tensor* a, float scalar);

/*
*   Returns a new tensor: result[i] = (a[i] <= scalar) ? 1.0 : 0.0
*/
Tensor* TensorLeScalar(Tensor* a, float scalar);

/*
*   Returns a new tensor: result[i] = (a[i] && b[i]) ? 1.0 : 0.0 (logical AND)
*/
Tensor* TensorAnd(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result[i] = (a[i] || b[i]) ? 1.0 : 0.0 (logical OR)
*/
Tensor* TensorOr(Tensor* a, Tensor* b);

/*
*   Returns a new tensor: result[i] = (a[i] XOR b[i]) ? 1.0 : 0.0 (logical XOR)
*/
Tensor* TensorXor(Tensor* a, Tensor* b);

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
*   Returns a new tensor with softmax applied along specified dimension
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