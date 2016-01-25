# MSplitGEMM
Large matrix multiplication in CUDA


# Preface/Disclaimer:
This repository is a set of algorithms that perform multiplication of very large matrices using the cuBLAS library in CUDA. These algorithms would be particularly useful for multiplication where the multiplicand and product matrices are too large to fit on the GPGPU.

Currently, this project has only been tested with square matrices, but it should be possible to perform multiplication of matrices of any legal input dimensions with a bit of fine tuning. 

# Introduction:
For normal matricies, using cuBLAS alone would be sufficient. For extremely large matrices, whose memory does not fit into the GPU global memory, an alternative method is to split the multiplicands into block matrices and perform the multiplication as shown in the figure below.

![Alt text](/Readme/blockMult.jpg?raw=true "Block Matrix Multiplication")

This repository contains four algorithms that implement the above equation in different ways. The algorithms here each have their own specific use cases, based on the advantages each one provides.  

# To Build/Run:
     make N will generate executable for algorithm N
     make 1 will generate Algorithm 1
     make 2 will generate Algorithm 2
     make 3 will generate Algorithm 3
     make 4 will generate Algorithm 4

Run the executable, providing a single integer as a command line argument that specifies the size of the square matrices. The application supports testing non-square matrices, but these operations are not yet supported in the algorithms.

    


# Algorithm 1: Generation of one output matrix at a time:    

This algorithm allocates memory for one submatrix of the first multiplicand and one submatrix of the second. Performs the single multiply and copies the memory back to the product matrix before moving on to the next computation. It uses the least memory of the algorithms, but generates the most memory transfer operations as we split the matrix into more submatrices.

    GPU Memory Usage =(1/numSubmatricies^2 + 2/numSubmatrices) * N^2 * sizeof(type)

![Alt text](/Readme/Alg1.jpg?raw=true "Algorithm 1 Visual Profile")

# Algorithm 2: Load one of the multiplicands entirely, and compute each row group together.

This algorithm allocates memory for one submatrix of the first multiplicand and the entire second multiplicand. Using this additional memory, we are able to launch all of the kernels for one row group at once. This allows for better performance at the cost of additional memory requirements.

    GPU Memory Usage =(1 + 2/numSubmatrices) * N^2 * sizeof(type)

![Alt text](/Readme/Alg2.jpg?raw=true "Algorithm 2 Visual Profile")

# Algorithm 3: Multithreaded streaming version of Algorithm 1 

Algorithm 3 is the first algorithm, but pipelined with CUDA streams, with each thread in the operation having a corresponding CUDA stream. This increases the performance of the first algorithm in most cases when the number of threads and streams is optimal for the number of submatrices and the size of the problem. A downside is that this algorithm uses more memory than the first on both the host and the GPU, because each thread or stream needs a context to work in.

    Gpu and Add. Host Mem Usage =(numStreams/numSubmatricies^2 + (1 + numStreams)/numSubmatrices) * N^2 * sizeof(type)

![Alt text](/Readme/Alg3.jpg?raw=true "Algorithm 3 Visual Profile")

# Algorithm 4: Streaming Version of Algorithm 1

Algorithm 4 does not use multiple threads on the CPU; instead, it exploits the natural paralellism of the CUDA streams to create the illusion of multiple CPU threads.

    Gpu and Add. Host Mem Usage =(numStreams/numSubmatricies^2 + (1 + numStreams)/numSubmatrices) * N^2 * sizeof(type)

![Alt text](/Readme/Alg4.jpg?raw=true "Algorithm 4 Visual Profile")
