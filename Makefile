CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -O2 -I$(INCLUDEDIR) -fopenmp
LDFLAGS = -fopenmp -lm
INCLUDEDIR = include
SRCDIR = src
BUILDDIR = build
SRCS = $(shell find $(SRCDIR) -name '*.c')
OBJS = $(SRCS:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
TARGET = ttl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILDDIR) $(TARGET)

depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ | sed 's|^\(.*\)\.o:|$(BUILDDIR)/\1.o:|' > .depend

-include .depend

.PHONY: all clean depend