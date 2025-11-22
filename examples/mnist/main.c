#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "../../include/io/file.h"
#include "../../include/tensor/tensor.h"
#include "mnist.h"

#define TRAIN_IMAGES "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
#define TRAIN_LABELS "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
#define TEST_IMAGES "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
#define TEST_LABELS "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
#define PATHMAX 256
#define DEFAULT_DATASET "./dataset"

void downloadDataset(char* dataset_path) {
    downloadFile(TRAIN_IMAGES, dataset_path, false);
    downloadFile(TRAIN_LABELS, dataset_path, false);
    downloadFile(TEST_IMAGES, dataset_path, false);
    downloadFile(TEST_LABELS, dataset_path, false);
}

void printUsage() {
    printf("TTL example for Machine Learning using the MNIST dataset.\n");
    printf("Usage:\n");
    printf("  ./mnist                 # use default dataset\n");
    printf("  ./mnist --download      # download into default dataset\n");
    printf("  ./mnist --download DIR  # download into DIR\n");
    exit(1);
}

/*
*   Used for debugging:
*   Example usage:
*       usize k = 5;
*       float* data = TensorData(test_images);
*       saveTensorAsPGM("digit.pgm", data + k * 28 * 28, 28, 28);
*/
void saveTensorAsPGM(const char* path, float* data, uint32_t rows, uint32_t cols) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P5\n%u %u\n255\n", cols, rows);
    
    for (uint32_t i = 0; i < rows * cols; i++) {
        unsigned char pixel = (unsigned char)(data[i] * 255.0f);
        fwrite(&pixel, 1, 1, f);
    }
    
    fclose(f);
}

Tensor* loadImages(char* filepath) {
    usize size;
    unsigned char* bytes = readGZip(filepath, &size);

    usize32 count = readBigEndian32(bytes + 4);
    usize32 rows = readBigEndian32(bytes + 8);
    usize32 cols = readBigEndian32(bytes + 12);
    unsigned char* pixels = bytes + 16;

    Tensor* images = TensorNew(SEQ(count, rows, cols), 3);

    #pragma omp parallel for
    for (usize32 k = 0; k < count; k++) {
        for (usize32 i = 0; i < rows; i++) {
            for (usize32 j = 0; j < cols; j++) {
                usize offset = k * rows * cols + i * cols + j;
                float val = pixels[offset] / 255.0f;  
                TensorSet(images, SEQ(k, i, j), val);
            }
        }
    }
    

    return images;
}

Tensor* loadLabels(char* filepath) {
    usize size;
    unsigned char* bytes = readGZip(filepath, &size);
    usize32 count = readBigEndian32(bytes + 4);
    unsigned char* labels = bytes + 8;

    Tensor* label_tensor = TensorNew(SEQ(count), 1);

    #pragma omp parallel for
    for (usize i = 0; i < count; i++) {
        float val = labels[i];
        TensorSet(label_tensor, SEQ(i), val);
    }

    return label_tensor;
}

bool parseArgs(int argc, char** argv, char** ds_path, bool* should_download) {
    *should_download = false;
    
    if (argc == 1) {
        *ds_path = DEFAULT_DATASET;
        return true;
    }
    if (argc == 2 && strcmp(argv[1], "--download") == 0) {
        *ds_path = DEFAULT_DATASET;
        *should_download = true;
        return true;
    }
    if (argc == 3 && strcmp(argv[1], "--download") == 0) {
        *ds_path = argv[2];
        *should_download = true;
        return true;
    }
    printUsage();
    return false;
}

float evaluate(MNISTNet* net, Tensor* images, Tensor* labels) {
    usize n = TensorDim(images, 0);
    usize correct = 0;

    // Flatten all images
    Tensor* x_flat = TensorReshape(images, SEQ(n, 784), 2);

    // Forward pass
    Tensor* pred = forward(net, x_flat);

    float* pred_data = TensorData(pred);
    float* label_data = TensorData(labels);

    for (usize i = 0; i < n; i++) {
        // Find argmax of prediction
        usize pred_label = 0;
        float max_val = pred_data[i * 10];
        for (usize j = 1; j < 10; j++) {
            if (pred_data[i * 10 + j] > max_val) {
                max_val = pred_data[i * 10 + j];
                pred_label = j;
            }
        }

        if (pred_label == (usize)label_data[i]) {
            correct++;
        }
    }

    TensorFree(x_flat);

    return (float)correct / (float)n;
}


MNISTNet* train(char* dataset_path) {
    char train_img_path[512]; snprintf(train_img_path, sizeof(train_img_path), "%s/train-images-idx3-ubyte.gz", dataset_path);
    char test_img_path[512]; snprintf(test_img_path, sizeof(test_img_path), "%s/t10k-images-idx3-ubyte.gz", dataset_path);
    char train_lbl_path[512]; snprintf(train_lbl_path, sizeof(train_lbl_path), "%s/train-labels-idx1-ubyte.gz", dataset_path);
    char test_lbl_path[512]; snprintf(test_lbl_path, sizeof(test_lbl_path), "%s/t10k-labels-idx1-ubyte.gz", dataset_path);

    Tensor* train_images = loadImages(train_img_path);
    Tensor* train_labels_raw = loadLabels(train_lbl_path);
    Tensor* train_labels = oneHotEncode(train_labels_raw, 10);
    TensorFree(train_labels_raw);

    Tensor* test_images = loadImages(test_img_path);
    Tensor* test_labels = loadLabels(test_lbl_path);
    
    MNISTNet* net = MNISTNetNew();

    usize n = TensorDim(train_images, 0);
    usize batch_size = 32;
    int epochs = 10;
    float lr = 0.1f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (usize i = 0; i < n; i += batch_size) {
            usize end = (i + batch_size < n) ? i + batch_size : n;
            usize actual_batch = end - i;

            Tensor* x_batch = TensorSlice(train_images, SLICES(RANGE(i, end), ALL, ALL));
            Tensor* y_batch = TensorSlice(train_labels, SLICES(RANGE(i, end), ALL));
            Tensor* x_flat = TensorReshape(x_batch, SEQ(actual_batch, 784), 2);

            forward(net, x_flat);
            backward(net, x_flat, y_batch);
            update(net, lr);

            TensorFree(x_batch);
            TensorFree(y_batch);
            TensorFree(x_flat);
        }

        float acc = evaluate(net, test_images, test_labels);
        printf("Epoch %d - Test accuracy: %.2f%%\n", epoch + 1, acc * 100);
    }

    TensorFree(train_images);
    TensorFree(train_labels);
    TensorFree(test_images);
    TensorFree(test_labels);
    
    return net;  
}

void test(MNISTNet* net, char* dataset_path) {
    char test_img_path[512]; 
    snprintf(test_img_path, sizeof(test_img_path), "%s/t10k-images-idx3-ubyte.gz", dataset_path);
    char test_lbl_path[512]; 
    snprintf(test_lbl_path, sizeof(test_lbl_path), "%s/t10k-labels-idx1-ubyte.gz", dataset_path);

    Tensor* test_images = loadImages(test_img_path);
    Tensor* test_labels = loadLabels(test_lbl_path);
    
    float acc = evaluate(net, test_images, test_labels);
    printf("Final test accuracy: %.2f%%\n", acc * 100);
    
    TensorFree(test_images);
    TensorFree(test_labels);
}

int main(int argc, char** argv) {
    char* dataset_path = NULL;
    bool should_download = false;
    
    if (!parseArgs(argc, argv, &dataset_path, &should_download)) {
        return 1;
    }
    
    if (should_download) {
        downloadDataset(dataset_path);
    }

    MNISTNet* net = train(dataset_path);
    test(net, dataset_path);
    MNISTNetFree(net);
    
    return 0;
}