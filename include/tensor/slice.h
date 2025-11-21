#ifndef TTL_SLICE_H
#define TTL_SLICE_H
#include "core/types.h"

typedef struct {
    isize start;
    isize stop;
    isize step;
    bool is_all;
} Slice;

/*
*   Slice a Tensor s.t. all elements are part of the new tensor.
*/
#define ALL (Slice){0, 0, 1, true}

/*
*   Slice a Tensor s.t. only element at index i is part of the view.
*/
#define AT(i) (Slice){i, i+1, 1, false} 

/*
*   Slice a Tensor to obtain view tensor[start:stop] at given dimension
*/
#define RANGE(start, stop) (Slice){start, stop, 1, false}

/*
*   Slice a Tensor to obtain view tensor[start:stop:step] at given dimension
*/
#define RANGE_STEP(start, stop, step) (Slice){start, stop, step, false}

/*
*   Slice a Tensor to obtain view tensor[from:length(dim)] at given dimension
*/
#define FROM(start) (Slice){start, ISIZE_MAX, 1, false}

/*
*   Slice a Tensor to obtain view tensor[0:stop] at given dimension
*/
#define TO(stop) (Slice){0, stop, 1, false}


/*
*   This macro is used to define an array of slices
*   Example:
*       //  tensor[0, :, 2:5]
*       Tensor* view = TensorSlice(t, SLICES(AT(0), ALL, RANGE(2, 5)));
*       //  tensor[:, ::2, -1]
*       Tensor* view = TensorSlice(t, SLICES(ALL, RANGE_STEP(0, ISIZE_MAX, 2), AT(-1)));
*/
#define SLICES(...) (Slice[]){__VA_ARGS__}, \
    (sizeof((Slice[]){__VA_ARGS__}) / sizeof(Slice))


#endif