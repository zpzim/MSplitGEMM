NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl
NVCC_SPEC_FLAGS = --default-stream per-thread
LD_FLAGS    = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl
EXE1	    = msplit-alg1
EXE2		= msplit-alg2
EXE3		= msplit-alg3-multi
EXE4		= msplit-alg3-seq
OBJ1	    = main1.o support.o
OBJ2		= main2.o support.o
OBJ3		= main3.o support.o
OBJ4		= main4.o support.o

default: $(EXE1)

1: $(EXE1)

2: $(EXE2)

3: $(EXE3)

4: $(EXE4)

main1.o: main1.cu kernel1.cu support.h
	$(NVCC) -c -o $@ main1.cu $(NVCC_FLAGS)

main2.o: main2.cu kernel2.cu support.h
	$(NVCC) -c -o $@ main2.cu $(NVCC_FLAGS)

main3.o: main3.cu kernel3.cu support.h
	$(NVCC) -c -o $@ main3.cu $(NVCC_FLAGS) $(NVCC_SPEC_FLAGS)

main4.o: main4.cu kernel4.cu support.h
	$(NVCC) -c -o $@ main4.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE1): $(OBJ1)
	$(NVCC) $(OBJ1) -o $(EXE1) $(LD_FLAGS)

$(EXE2): $(OBJ2)
	$(NVCC) $(OBJ2) -o $(EXE2) $(LD_FLAGS)

$(EXE3): $(OBJ3)
	$(NVCC) $(OBJ3) -o $(EXE3) $(LD_FLAGS)

$(EXE4): $(OBJ4)
	$(NVCC) $(OBJ4) -o $(EXE4) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE1) $(EXE2) $(EXE3) $(EXE4)
