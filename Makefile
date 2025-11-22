CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -O2 -I$(INCLUDEDIR) -fopenmp
LDFLAGS = -fopenmp
LIBS = -lm
INCLUDEDIR = include
SRCDIR = src
BUILDDIR = build

# Only compile library sources (tensor, random, etc.) — no main.c
SRCS = $(shell find $(SRCDIR) -name '*.c' ! -name 'main.c')
OBJS = $(SRCS:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)

# BLAS detection
ifdef USE_BLAS
    CFLAGS += -DUSE_BLAS
    LIBS += $(or $(BLAS_LIB),-lopenblas)
else
    BLAS_PKG := $(shell pkg-config --libs openblas 2>/dev/null || pkg-config --libs blas 2>/dev/null)
    ifneq ($(BLAS_PKG),)
        CFLAGS += -DUSE_BLAS
        LIBS += $(BLAS_PKG)
    else
        UNAME_S := $(shell uname -s)
        ifeq ($(UNAME_S),Darwin)
            CFLAGS += -DUSE_BLAS -DUSE_ACCELERATE
            LIBS += -framework Accelerate
        else ifeq ($(UNAME_S),Linux)
            ifneq ($(wildcard /usr/lib/x86_64-linux-gnu/libopenblas.so /usr/lib/libopenblas.so),)
                CFLAGS += -DUSE_BLAS
                LIBS += -lopenblas
            else ifneq ($(wildcard /usr/lib/libblas.so),)
                CFLAGS += -DUSE_BLAS
                LIBS += -lblas
            endif
        endif
    endif
endif

# Build library objects only
all: $(OBJS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# MNIST example
MNIST_DIR = examples/mnist
MNIST_TARGET = $(MNIST_DIR)/mnist
MNIST_LIBS = $(LIBS) -lz

mnist: $(OBJS)
	$(CC) $(CFLAGS) $(MNIST_DIR)/main.c $(OBJS) -o $(MNIST_TARGET) $(LDFLAGS) $(MNIST_LIBS)

run-mnist:
	$(MNIST_DIR)/mnist --download 

# Cleanup
clean:
	rm -rf $(BUILDDIR)

clean-mnist:
	rm -f $(MNIST_TARGET)

clean-all: clean clean-mnist

.PHONY: all clean clean-mnist clean-all mnist