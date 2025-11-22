#include <stdlib.h>
#include "../../include/tensor/tensor.h"


typedef struct {
    Tensor* W1;   // shape [784, 128]
    Tensor* b1;   // shape [128]
    
    Tensor* W2;   // shape [128, 10]
    Tensor* b2;   // shape [10]
    
    // Cache for backward pass
    Tensor* z1;   // pre-activation layer 1
    Tensor* a1;   // post-ReLU layer 1
    Tensor* z2;   // pre-softmax
    Tensor* a2;   // output (softmax)
    
    // Gradients
    Tensor* dW1;
    Tensor* db1;
    Tensor* dW2;
    Tensor* db2;
} MNISTNet;


MNISTNet* MNISTNetNew() {
    MNISTNet* net = malloc(sizeof(MNISTNet));
    float std1 = sqrtf(2.0f / 784);
    net->W1 = TensorRandN(SEQ(784, 128), 2, 0.0f, std1);
    net->b1 = TensorZero(SEQ(128), 1);

    float std2 = sqrtf(2.0f / 128);
    net->W2 = TensorRandN(SEQ(128, 10), 2, 0.0f, std2);
    net->b2 = TensorZero(SEQ(10), 1);

    net->dW1 = TensorZero(SEQ(784, 128), 2);
    net->db1 = TensorZero(SEQ(128), 1);
    net->dW2 = TensorZero(SEQ(128, 10), 2);
    net->db2 = TensorZero(SEQ(10), 1);
    
    net->z1 = net->a1 = net->z2 = net->a2 = NULL;
    
    return net;
}

void MNISTNetFree(MNISTNet* net) {
    TensorFree(net->W1);
    TensorFree(net->b1);
    TensorFree(net->W2);
    TensorFree(net->b2);
    TensorFree(net->dW1);
    TensorFree(net->db1);
    TensorFree(net->dW2);
    TensorFree(net->db2);
    if (net->z1) TensorFree(net->z1);
    if (net->a1) TensorFree(net->a1);
    if (net->z2) TensorFree(net->z2);
    if (net->a2) TensorFree(net->a2);
    free(net);
}

void addBias_(Tensor* z, Tensor* b) {
    usize batch = TensorDim(z, 0);
    usize features = TensorDim(z, 1);
    float* zd = TensorData(z);
    float* bd = TensorData(b);
    
    for (usize i = 0; i < batch; i++) {
        for (usize j = 0; j < features; j++) {
            zd[i * features + j] += bd[j];
        }
    }
}

Tensor* forward(MNISTNet* net, Tensor* x) {
    // Free previous batch's cached values
    if (net->z1) TensorFree(net->z1);
    if (net->a1) TensorFree(net->a1);
    if (net->z2) TensorFree(net->z2);
    if (net->a2) TensorFree(net->a2);

    // Layer 1: z1 = x @ W1 + b1
    net->z1 = TensorMatMul(x, net->W1);  // [batch, 128]
    // Add bias (need to broadcast b1 across batch)
    addBias_(net->z1, net->b1);
    
    // ReLU
    net->a1 = TensorRelu(net->z1);  // [batch, 128]
    
    // Layer 2: z2 = a1 @ W2 + b2
    net->z2 = TensorMatMul(net->a1, net->W2);  // [batch, 10]
    addBias_(net->z2, net->b2);
    
    // Softmax
    net->a2 = TensorSoftmax(net->z2, 1);  // softmax along dim 1
    
    return net->a2;
}

void backward(MNISTNet* net, Tensor* x, Tensor* y) {
    usize batch = TensorDim(x, 0);
    
    // Output layer gradient: dz2 = a2 - y (for softmax + cross-entropy)
    Tensor* dz2 = TensorSub(net->a2, y);  // [batch, 10]
    
    // dW2 = a1.T @ dz2
    Tensor* a1T = TensorTranspose(net->a1);  // [128, batch]
    TensorFree(net->dW2);
    net->dW2 = TensorMatMul(a1T, dz2);  // [128, 10]
    TensorFree(a1T);
    
    // db2 = sum(dz2, dim=0)
    TensorFree(net->db2);
    net->db2 = TensorSumDim(dz2, 0, false);  // [10]
    
    // da1 = dz2 @ W2.T
    Tensor* W2T = TensorTranspose(net->W2);  // [10, 128]
    Tensor* da1 = TensorMatMul(dz2, W2T);  // [batch, 128]
    TensorFree(W2T);
    
    // dz1 = da1 * relu_derivative(z1)
    // relu_derivative: 1 if z1 > 0, else 0
    Tensor* dz1 = TensorCopy(da1);
    float* dz1d = TensorData(dz1);
    float* z1d = TensorData(net->z1);
    usize size = TensorSize(dz1);
    for (usize i = 0; i < size; i++) {
        if (z1d[i] <= 0) dz1d[i] = 0;
    }
    TensorFree(da1);
    
    // dW1 = x.T @ dz1
    Tensor* xT = TensorTranspose(x);  // [784, batch]
    TensorFree(net->dW1);
    net->dW1 = TensorMatMul(xT, dz1);  // [784, 128]
    TensorFree(xT);
    
    // db1 = sum(dz1, dim=0)
    TensorFree(net->db1);
    net->db1 = TensorSumDim(dz1, 0, false);  // [128]
    
    TensorFree(dz2);
    TensorFree(dz1);
    
    // Scale gradients by batch size
    TensorDivScalar_(net->dW1, (float)batch);
    TensorDivScalar_(net->db1, (float)batch);
    TensorDivScalar_(net->dW2, (float)batch);
    TensorDivScalar_(net->db2, (float)batch);
}

void update(MNISTNet* net, float lr) {
    // W = W - lr * dW
    Tensor* step;
    
    step = TensorMulScalar(net->dW1, lr);
    TensorSub_(net->W1, step);
    TensorFree(step);
    
    step = TensorMulScalar(net->db1, lr);
    TensorSub_(net->b1, step);
    TensorFree(step);
    
    step = TensorMulScalar(net->dW2, lr);
    TensorSub_(net->W2, step);
    TensorFree(step);
    
    step = TensorMulScalar(net->db2, lr);
    TensorSub_(net->b2, step);
    TensorFree(step);
}

Tensor* oneHotEncode(Tensor* labels, usize num_classes) {
    usize count = TensorDim(labels, 0);
    Tensor* onehot = TensorZero(SEQ(count, num_classes), 2);
    float* label_data = TensorData(labels);
    
    for (usize i = 0; i < count; i++) {
        usize label = (usize)label_data[i];
        TensorSet(onehot, SEQ(i, label), 1.0f);
    }
    
    return onehot;
}