# Kernel module Makefile for onyx_sched.ko
#
# Build on Jetson (native):
#   make
#
# Cross-compile for Jetson from x86:
#   make CROSS_COMPILE=aarch64-linux-gnu- KERNEL_DIR=/path/to/l4t-kernel
#
# Install:
#   sudo insmod onyx_sched.ko
#
# Uninstall:
#   sudo rmmod onyx_sched

obj-m := onyx_sched.o

# Kernel source directory (auto-detect or override)
KERNEL_DIR ?= /lib/modules/$(shell uname -r)/build

# Cross-compilation support
CROSS_COMPILE ?=
ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/')

all:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) modules

clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean

install:
	sudo insmod onyx_sched.ko
	@echo "onyx_sched loaded — /dev/onyx_sched available"

uninstall:
	sudo rmmod onyx_sched
	@echo "onyx_sched unloaded"

.PHONY: all clean install uninstall
