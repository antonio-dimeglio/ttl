CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -O2 -I$(INCLUDEDIR) -fopenmp
LDFLAGS = -fopenmp
LIBS = -lm
INCLUDEDIR = include
SRCDIR = src
BUILDDIR = build
SRCS = $(shell find $(SRCDIR) -name '*.c')
OBJS = $(SRCS:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
TARGET = ttl

# BLAS detection: tries pkg-config, then platform-specific paths, then falls back to naive
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

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILDDIR) $(TARGET)

depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ | sed 's|^\(.*\)\.o:|$(BUILDDIR)/\1.o:|' > .depend

-include .depend

.PHONY: all clean depend