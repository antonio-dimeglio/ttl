#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define TRAIN_IMAGES "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
#define TRAIN_LABELS "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
#define TEST_IMAGES "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
#define TEST_LABELS "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
#define PATHMAX 256
#define DEFAULT_DATASET "./dataset"

void fetchFile(char* url, char* save_location) {
    
}

void downloadDataset(char* dataset_path) {
    fetchFile(TRAIN_IMAGES, dataset_path);
    fetchFile(TRAIN_LABELS, dataset_path);
    fetchFile(TEST_IMAGES, dataset_path);
    fetchFile(TEST_LABELS, dataset_path);
}

void printUsage() {
    printf("TTL example for Machine Learning using the MNIST dataset.\n");
    printf("Usage:\n");
    printf("  ./mnist                 # use default dataset\n");
    printf("  ./mnist --download      # download into default dataset\n");
    printf("  ./mnist --download DIR  # download into DIR\n");
    exit(1);
}

bool parseArgs(int argc, char** argv, char** ds_path) {
    if (argc == 1) {
        *ds_path = DEFAULT_DATASET;
        return true;
    }

    if (argc == 2) {
        if (strcmp(argv[1], "--download") == 0) {
            *ds_path = DEFAULT_DATASET;
            return true;
        } else {
            printUsage();
            return false;
        }
    }

    if (argc == 3) {
        if (strcmp(argv[1], "--download") == 0) {
            *ds_path = argv[2];
            return true;
        } else {
            printUsage();
            return false;
        }
    }

    printUsage();
    return false;
}

int main(int argc, char** argv) {
    char* dataset_path = NULL;
    bool download_dataset = parseArgs(argc, argv, &dataset_path);
    if (download_dataset) {
        downloadDataset(dataset_path);
    }

    return 0;
}