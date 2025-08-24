TARGET = main

SRC = main.cpp cpu_attention.cpp gpu_attention.cu cpu_autoregressive_attention.cpp gpu_autoregressive_attention.cu paged_attention.cu

NVCC = nvcc

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^

debug: $(SRC)
	$(NVCC) -g -G -O0 -lineinfo -Xcompiler "-g -O0 -fno-omit-frame-pointer" -o $(TARGET) $^ 

.PHONY: clean debug
clean:
	rm $(TARGET)

