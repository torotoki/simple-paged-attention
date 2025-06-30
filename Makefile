TARGET = main

SRC = main.cpp #gpu_scan.cu

NVCC = nvcc

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^

clean:
	rm $(TARGET)

