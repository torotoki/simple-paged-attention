TARGET = main

SRC = main.cpp cpu_attention.cpp gpu_attention.cu cpu_autoregressive_attention.cpp gpu_autoregressive_attention.cu

NVCC = nvcc

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^

clean:
	rm $(TARGET)

