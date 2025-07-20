TARGET = main

SRC = main.cpp cpu_attention.cpp gpu_attention.cu cpu_autoregressive_attention.cpp

NVCC = nvcc

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^

clean:
	rm $(TARGET)

