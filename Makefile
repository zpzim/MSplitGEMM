NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  -lcublas_static -lculibos -lcudart_static -lpthread -ldl
NVCC_SPEC_FLAGS = --default-stream per-thread
LD_FLAGS    = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl
EXE1	    = msplit-alg1
EXE2		= msplit-alg2
EXE3		= msplit-alg3-multi
EXE4		= msplit-alg3-seq
OBJ1	    = main1.o support.o common.o
OBJ2		= main2.o support.o common.o
OBJ3		= main3.o support.o common.o
OBJ4		= main4.o support.o common.o

default: $(EXE1)

1: $(EXE1)

2: $(EXE2)

3: $(EXE3)

4: $(EXE4)

main1.o: main.cu kernel1.cu support.h common.h
	$(NVCC) -c -o $@ main.cu -DKERNEL1=0 $(NVCC_FLAGS)

main2.o: main.cu kernel2.cu support.h common.h
	$(NVCC) -c -o $@ main.cu -DKERNEL2=0 $(NVCC_FLAGS)

main3.o: main.cu kernel3.cu support.h common.h
	$(NVCC) -c -o $@ main.cu -DKERNEL3=0 $(NVCC_FLAGS) $(NVCC_SPEC_FLAGS)

main4.o: main.cu kernel4.cu support.h common.h
	$(NVCC) -c -o $@ main.cu -DKERNEL4=0 $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

common.o: common.h common.cu
	$(NVCC) -c -o $@ common.cu $(NVCC_FLAGS)

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
