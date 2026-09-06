void run()
{
    constexpr int N = 21, NRHS = 9, P = 4, ARCH = 900;
    constexpr int nt = N + operators::npoly(P);

    const int nnz = nstencils * N;

    using T      = double
    using solver = operators::GESV<T, nt, NRHS, ARCH>

    std::vector<double> x, y;
    auto nstencils = read_coordinates("filename.txt", x, y);

    T *x_d, *y_d, *xc_d, *yc_d;
    int *ja_d, *info_d;

    CUDA_CHECK(cudaMalloc(&x_d,    sizeof(T) * nstencils));
    CUDA_CHECK(cudaMalloc(&y_d,    sizeof(T) * nstencils));
    CUDA_CHECK(cudaMalloc(&xc_d,   sizeof(T) * NRHS));
    CUDA_CHECK(cudaMalloc(&yc_d,   sizeof(T) * NRHS));
    CUDA_CHECK(cudaMalloc(&ja_d,   sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&info_d, sizeof(int) * nstencils));

    CUDA_CHECK(cudaMemcpy(x_d,  x.data(),  sizeof(T) * nstencils, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d,  y.data(),  sizeof(T) * nstencils, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(xc_d, xc.data(), sizeof(T) * NRHS,      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(yc_d, yc.data(), sizeof(T) * NRHS,      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ja_d, ja.data(), sizeof(int) * nnz,     cudaMemcpyHostToDevice));

    // A is T* const* : a DEVICE array of NRHS device pointers, one CSR values
    // array per operator. Building the table on the host and copying it over is
    // the part that is easy to get wrong.
    std::array<T*, NRHS> A_h;
    for (auto& p : A_h) {
        CUDA_CHECK(cudaMalloc(&p, sizeof(T) * nnz));
    }

    T** A_d;
    CUDA_CHECK(cudaMalloc(&A_d, sizeof(T*) * NRHS));
    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), sizeof(T*) * NRHS, cudaMemcpyHostToDevice));

    // --- launch configuration ------------------------------------------------

    const dim3 block{solver::blockDim};
    const unsigned int smem = ...;

    /* check attributes */ {
        auto kernel = &operators::assemble_interp<N, NRHS, P, ARCH, T>;

        // Mandatory above 48 KB, harmless below it
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem));

        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel)));
        printf("block=%u smem=%u regs=%d spill=%zu maxdyn=%d\n",
               block.x * block.y * block.z, smem, attr.numRegs,
               attr.localSizeBytes, attr.maxDynamicSharedSizeBytes);
    }


    operators::assemble_interp<N, NRHS, P, T>
        <<<nstencils, block, smem>>>(nstencils, x_d, y_d,
                                     ja_d, A_d,
                                     xc_d, yc_d,
                                     info_d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- status check --------------------------------------------------------

    std::vector<int> info(nstencils);

    CUDA_CHECK(cudaMemcpy(info.data(), info_d, sizeof(int) * nstencils,
                          cudaMemcpyDevicetoHost));


}

