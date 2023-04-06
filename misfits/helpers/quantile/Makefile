# Define variables
MODULE_NAME = smart_quantile
CYTHON_SRC = $(MODULE_NAME).pyx

# Define targets
all: $(CYTHON_SRC)
	python setup_$(MODULE_NAME).py build_ext --inplace

clean:
	rm -rf build
	rm *.c *.so
	rm -rf __pycache__

.PHONY: all clean

