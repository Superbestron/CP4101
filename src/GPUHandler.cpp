#include "../common/includes/GPUHandler.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>
#include <iostream>
#define DEBUG

#include "../common/includes/helpers.h"
#include "../common/includes/rocsparseUtility.h"

// #define PRINT_MATRIX
// #define FUNCTIONAL_VERIFICATION

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if (stat != hipSuccess)                                                \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
            exit(-1);                                                          \
        }                                                                      \
    }

#define ROCSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if (stat != rocsparse_status_success)                                        \
        {                                                                            \
            std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
            exit(-1);                                                                \
        }                                                                            \
    }

#define ROCBLAS_CHECK(stat)                                                          \
    {                                                                                \
        if (stat != rocsparse_status_success)                                        \
        {                                                                            \
            std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
            exit(-1);                                                                \
        }                                                                            \
    }

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                  \
    if (status != rocblas_status_success)             \
    {                                                 \
        fprintf(stderr, "rocBLAS error: ");           \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif

inline double GPUHandler::utils_time_us(void) {
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  return (static_cast<double>(duration));
};

void GPUHandler::setupData(
    const SparseMatrix &sparse_matrix,
    vector<float> &feature_matrix,
    const int &n,
    const float &ALPHA,
    const float &BETA) {

  int m = sparse_matrix.M;
  int k = sparse_matrix.K;
  int nnz_A = sparse_matrix.nnz;
  int device_id = 0;

  hC = std::vector<data_type>(m * n, static_cast<data_type>(0));

  HIP_CHECK(hipGetDevice(&device_id));

  HIP_CHECK(hipMalloc((void **) &dAptr, sizeof(index_type) * (m + 1)));
  HIP_CHECK(hipMalloc((void **) &dAcol, sizeof(index_type) * nnz_A));
  HIP_CHECK(hipMalloc((void **) &dAval, sizeof(data_type) * nnz_A));
  // HIP_CHECK(hipMalloc((void **)&dB, sizeof(data_type) * k * n));
  HIP_CHECK(hipMalloc((void **) &dspmmC, sizeof(data_type) * m * n));

  spmmA_moving_time_cpu = utils_time_us();
  HIP_CHECK(hipMemcpy(dAptr, sparse_matrix.CSRRowPtr.data(), sizeof(index_type) * (m + 1),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dAcol, sparse_matrix.CSRColIndex.data(), sizeof(index_type) * nnz_A,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dAval, sparse_matrix.CSRVal.data(), sizeof(data_type) * nnz_A,
                      hipMemcpyHostToDevice));

  // HIP_CHECK(hipMemcpy(dAptr, sparse_matrix.CSCColPtr.data(), sizeof(index_type) * (k + 1),
  //                        hipMemcpyHostToDevice));
  // HIP_CHECK(hipMemcpy(dAcol, sparse_matrix.CSCRowIndex.data(), sizeof(index_type) * nnz_A,
  //                        hipMemcpyHostToDevice));
  // HIP_CHECK(hipMemcpy(dAval, sparse_matrix.CSCVal.data(), sizeof(data_type) * nnz_A,
  //                        hipMemcpyHostToDevice));

  HIP_CHECK(hipDeviceSynchronize());
  spmmA_moving_time_cpu = (utils_time_us() - spmmA_moving_time_cpu) / (1e3);

  spmmCin_moving_time_cpu = utils_time_us();
  HIP_CHECK(hipMemcpy(dspmmC, hC.data(), sizeof(data_type) * m * n,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
  spmmCin_moving_time_cpu = (utils_time_us() - spmmCin_moving_time_cpu) / (1e3);

  cinThroughput = (sizeof(data_type) * m * n / 1e9) / (spmmCin_moving_time_cpu / 1000);
  cinSize = sizeof(data_type) * m * n / 1e6;
}

void GPUHandler::executeSparseMultiplication(
    const SparseMatrix &sparse_matrix,
    const vector<float> &feature_matrix,
    const int &n,
    const float &ALPHA,
    const float &BETA) {

  int m = sparse_matrix.M;
  int k = sparse_matrix.K;
  int nnz_A = sparse_matrix.nnz;

  // std::vector<data_type> hC(m * n, static_cast<data_type>(0));
  data_type halpha = static_cast<data_type>(ALPHA);
  data_type hbeta = static_cast<data_type>(BETA);

#ifdef PRINT_MATRIX
  std::cout << std::endl;
    std::cout << "Sparse Matrix Row Pointers: " << std::endl;
    int i = 0;
    for (auto& a_ptr : sparse_matrix.CSRRowPtr) {
      std::cout << a_ptr << ",";
      i++;
      if (i == 10)
        break;
    }
    std::cout << std::endl;
    std::cout << "Sparse Matrix Column Indices: " << std::endl;
    i = 0;
    for (auto& a_col : sparse_matrix.CSRColIndex) {
      std::cout << a_col << ",";
      i++;
      if (i == 10)
        break;
    }
    std::cout << std::endl;
    std::cout << "Sparse Matrix Values: " << std::endl;
    i = 0;
    for (auto& a_val : sparse_matrix.CSRVal) {
      std::cout << a_val << ",";
      i++;
      if (i == 10)
        break;
    }
    std::cout << std::endl;
    std::cout << "Dense Matrix Values: " << std::endl;
    i = 0;
    for (auto& b_val : feature_matrix) {
      std::cout << b_val << ",";
      i++;
      if (i == 10)
        break;
    }
    std::cout << std::endl;
#endif

  // rocSPARSE handle
  rocsparse_handle handle;
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

  hipDeviceProp_t devProp;
  int device_id = 0;

  HIP_CHECK(hipGetDevice(&device_id));
  HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
  // std::cout << "Device: " << devProp.name << std::endl;
  // std::cout.precision(2);
  // std::cout.setf(std::ios::fixed);
  // std::cout.setf(std::ios::left);
  // std::cout << std::endl;

  // Offload data to device
  data_type *dB = NULL;
  HIP_CHECK(hipMalloc((void **) &dB, sizeof(data_type) * k * n));

  spmmB_moving_time_cpu = utils_time_us();
  HIP_CHECK(hipMemcpy(dB, feature_matrix.data(), sizeof(data_type) * k * n,
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipDeviceSynchronize());
  spmmB_moving_time_cpu = (utils_time_us() - spmmB_moving_time_cpu) / 1e3;


  // Types
  rocsparse_indextype itype = utils_indextype<index_type>();
  rocsparse_indextype jtype = utils_indextype<index_type>();
  rocsparse_datatype ttype = utils_datatype<data_type>();

  // Create descriptors
  rocsparse_spmat_descr A;
  rocsparse_dnmat_descr B;
  rocsparse_dnmat_descr C;
  ROCSPARSE_CHECK(rocsparse_create_csr_descr(&A,
                                             m,
                                             k,
                                             nnz_A,
                                             dAptr,
                                             dAcol,
                                             dAval,
                                             itype,
                                             jtype,
                                             rocsparse_index_base_zero,
                                             ttype));
  ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&B, k, n, n, dB, ttype, rocsparse_order_row));
  ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&C, m, n, n, dspmmC, ttype, rocsparse_order_row));

  // ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&B, k, n, ncol_B, dB, ttype, rocsparse_order_row));
  // ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&C, m, n, n, dC, ttype, rocsparse_order_row));
  // ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&B, k, n, k, dB, ttype, rocsparse_order_column));
  // ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&C, m, n, m, dspmmC, ttype, rocsparse_order_column));

  // rocsparse_mat_descr descr_A;
  // ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));

  size_t buffer_size;
  spmm_kernel_exec_time = utils_time_us();

  // ROCSPARSE_CHECK(rocsparse_scsrmm(handle,
  //        rocsparse_operation_none,
  //        rocsparse_operation_none,
  //        m,
  //        n,
  //        k,
  //        nnz_A,
  //        &halpha,
  //        A,
  //        dAval,
  //        dAptr,
  //        dAcol,
  //        dB,
  //        k,
  //        &beta,
  //        C,
  //        m));

  ROCSPARSE_CHECK(rocsparse_spmm(handle,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 &halpha,
                                 A,
                                 B,
                                 &hbeta,
                                 C,
                                 ttype,
                                 rocsparse_spmm_alg_default,
                                 rocsparse_spmm_stage_buffer_size,
                                 &buffer_size,
                                 nullptr));

  // Allocate buffer
  void *buffer;
  hipMalloc(&buffer, buffer_size);

  ROCSPARSE_CHECK(rocsparse_spmm(handle,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 &halpha,
                                 A,
                                 B,
                                 &hbeta,
                                 C,
                                 ttype,
                                 rocsparse_spmm_alg_default,
                                 rocsparse_spmm_stage_preprocess,
                                 &buffer_size,
                                 buffer));

  ROCSPARSE_CHECK(rocsparse_spmm(handle,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 &halpha,
                                 A,
                                 B,
                                 &hbeta,
                                 C,
                                 ttype,
                                 rocsparse_spmm_alg_default,
                                 rocsparse_spmm_stage_compute,
                                 &buffer_size,
                                 buffer));
  HIP_CHECK(hipDeviceSynchronize());

  spmm_kernel_exec_time = (utils_time_us() - spmm_kernel_exec_time) / 1e3;
#ifdef PRINT_MATRIX

  HIP_CHECK(hipMemcpy(hC.data(), dspmmC, sizeof(data_type) * m * n,
              hipMemcpyDeviceToHost));


    std::cout << "Output after SpMM:" << std::endl;
    i = 0;
    for (auto& c_val : hC) {
        i++;
        if (i < 100 && c_val != 0)
            std::cout << std::setw(10) << std::fixed
                      << std::setprecision(std::numeric_limits<data_type>::digits10)
                      << c_val << ",";
        if (i == 100)
         break;
    }
    std::cout << std::endl;

#endif

  // Clear up on device
  // HIP_CHECK(hipFree(dAptr));
  // HIP_CHECK(hipFree(dAcol));
  // HIP_CHECK(hipFree(dAval));
  HIP_CHECK(hipFree(dB));
  // HIP_CHECK(hipFree(dC));

  ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(A));
  ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(B));
  ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(C));

  float gflops = 2.0 * n * (nnz_A + m) / 1e9 // convert to GB
  ;
  // std::cout << "GFLOP: " << gflops << std::endl;

#ifdef FUNCTIONAL_VERIFICATION
  // CPU Validation
    std::vector<vector<data_type>> golden_output(m, vector<data_type>(n));
    std::vector<vector<data_type>> B_mat(k, vector<data_type>(n));

    //converting dense matrix represented in vector format to matrix format
    int id = 0;
    for (int i = 0 ; i < k; i++){
        for(int j = 0; j < n; j++){
         B_mat[i][j] = feature_matrix[id++];
        }
    }

    // CPU results
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            golden_output[i][j] = hbeta * golden_output[i][j];
            for(int k = sparse_matrix.CSRRowPtr[i]; k < sparse_matrix.CSRRowPtr[i + 1]; ++k){
                      golden_output[i][j] += halpha * sparse_matrix.CSRVal[k] * B_mat[sparse_matrix.CSRColIndex[k]][j];
          }
        }
    }

    // std::cout << "Golden Output:" << std::endl;
    // i = 0;

    // for (auto& c_val : golden_output[0]) {
    //     std::cout << std::setw(10) << std::fixed
    //       << std::setprecision(std::numeric_limits<data_type>::digits10)
    //       << c_val << ",";
    //     i++;
    //     if (i == 100)
    //             break;
    // }
    // std::cout << std::endl;

    std::cout << "Golden Output:" << std::endl;
    i = 0;
    for (auto& c_row : golden_output) {
        for (auto& c_val : c_row) {
            i++;
            if (i < 100 && c_val != 0)
             std::cout << std::setw(10) << std::fixed
                              << std::setprecision(std::numeric_limits<data_type>::digits10)
                              << c_val << ",";
            if (i == 100)
              break;
        }
    }
    std::cout << std::endl;


#endif

}

void GPUHandler::verifyGPUResults(vector<float> &mat_C_cpu, int M, int N) {
}

vector<data_type> GPUHandler::executeDenseMultiplication(
    std::vector<data_type> &l1_weight_matrix,
    int num_nodes,
    int feature_size,
    int l1_size) {
  rocblas_status rstatus = rocblas_status_success;

#ifdef PRINT_MATRIX
  std::cout << std::endl;
    int i = 0;

    std::cout << "Dense Matrix Values (B): " << std::endl;
    i = 0;
    for (auto& b_val : l1_weight_matrix) {
      std::cout << b_val << ",";
      i++;
      if (i == 10)
        break;
    }
    std::cout << std::endl;
#endif

  data_type halpha = static_cast<data_type>(1);
  data_type hbeta = static_cast<data_type>(0);

  const rocblas_operation transA = rocblas_operation_none;
  const rocblas_operation transB = rocblas_operation_none;

  int lda, ldb, ldc, sizeA, sizeB, sizeC;
  int strideA1, strideA2, strideB1, strideB2;

  if (transA == rocblas_operation_none) {
    lda = num_nodes;
    sizeA = feature_size * lda;
    strideA1 = 1;
    strideA2 = lda;
  } else {
    lda = feature_size;
    sizeA = num_nodes * lda;
    strideA1 = lda;
    strideA2 = 1;
  }
  if (transB == rocblas_operation_none) {
    ldb = feature_size;
    sizeB = l1_size * ldb;
    strideB1 = 1;
    strideB2 = ldb;
  } else {
    ldb = l1_size;
    sizeB = feature_size * ldb;
    strideB1 = ldb;
    strideB2 = 1;
  }
  ldc = num_nodes;
  sizeC = l1_size * ldc;

  // using rocblas API
  rocblas_handle handle;
  rstatus = rocblas_create_handle(&handle);
  CHECK_ROCBLAS_STATUS(rstatus);

  // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
  std::vector<data_type> hgemmC(sizeC);
  std::vector<data_type> hGold(sizeC);

#ifdef PRINT_MATRIX
  std::cout << "Output before Dense MM:" << std::endl;
    i = 0;
    for (auto& c_val : hC) {
      std::cout << std::setw(10) << std::fixed
                << std::setprecision(std::numeric_limits<data_type>::digits10)
                << c_val << ",";
      i++;
      if (i == 100)
        break;
    }
    std::cout << std::endl;
#endif

  hGold = hgemmC;
  {
    // allocate memory on device
    // helpers::DeviceVector<data_type> dA_(sizeA);
    helpers::DeviceVector<data_type> dgemmB(sizeB);
    helpers::DeviceVector<data_type> dgemmC(sizeC);

    gemmBC_move_compute_and_C_move_time = utils_time_us();
    CHECK_HIP_ERROR(hipMemcpy(
        dgemmB, static_cast<void *>(l1_weight_matrix.data()), sizeof(data_type) * sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dgemmC, static_cast<void *>(hgemmC.data()), sizeof(data_type) * sizeC, hipMemcpyHostToDevice));

    // enable passing alpha parameter from pointer to host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    // asynchronous calculation on device, returns before finished calculations
    rstatus = rocblas_sgemm(
        handle,
        transA,
        transB,
        num_nodes,
        l1_size,
        feature_size,
        &halpha,
        dspmmC,
        lda,
        dgemmB,
        ldb,
        &hbeta,
        dgemmC,
        ldc);

    // check that calculation was launched correctly on device, not that result
    // was computed yet
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch device memory results, automatically blocked until results ready
    CHECK_HIP_ERROR(hipMemcpy(hgemmC.data(), dgemmC, sizeof(data_type) * sizeC, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    gemmBC_move_compute_and_C_move_time = (utils_time_us() - gemmBC_move_compute_and_C_move_time) / 1e3;

  } // release device memory via helpers::DeviceVector destructors

#ifdef PRINT_MATRIX
  std::cout << "Output after Dense MM:" << std::endl;
    i = 0;
    for (auto& c_val : hgemmC) {
      std::cout << std::setw(10) << std::fixed
                << std::setprecision(std::numeric_limits<data_type>::digits10)
                << c_val << ",";
      i++;
      if (i == 100)
        break;
    }
    std::cout << std::endl;
#endif

  // #ifdef FUNCTIONAL_VERIFICATION
  // // calculate gold standard using CPU
  // helpers::matMatMult<data_type>(halpha,
  //                               hbeta,
  //                               num_nodes,
  //                               l1_size,
  //                               feature_size,
  //                               AX.data(),
  //                               strideA1,
  //                               strideA2,
  //                               l1_weight_matrix.data(),
  //                               strideB1,
  //                               strideB2,
  //                               hGold.data(),
  //                               1,
  //                               ldc);

  // #ifdef PRINT_MATRIX
  // std::cout << "Golden output:" << std::endl;
  // i = 0;
  // for (auto& c_val : hGold) {
  //   std::cout << std::setw(10) << std::fixed
  //             << std::setprecision(std::numeric_limits<data_type>::digits10)
  //             << c_val << ",";
  //   i++;
  //   if (i == 100)
  //     break;
  // }
  // std::cout << std::endl;
  // #endif
  // data_type maxRelativeError = (data_type)helpers::maxRelativeError(hC, hGold);
  // data_type eps              = std::numeric_limits<data_type>::epsilon();
  // data_type tolerance        = 50;
  // if(maxRelativeError > eps * tolerance)
  // {
  //     std::cout << "FAIL";
  // }
  // else
  // {
  //     std::cout << "PASS";
  // }
  // std::cout << ": max. relative err. = " << maxRelativeError << std::endl;
  // #endif

  rstatus = rocblas_destroy_handle(handle);
  CHECK_ROCBLAS_STATUS(rstatus);
  return hgemmC;
}

__global__ void redistribute(float *c1,
                             float *c2,
                             float *c3,
                             float *c4,
                             float *c,
                             int fpga_chunk_size,
                             int fpga_column_size) {
  // asm volatile ("buffer_wbinvl1_vol");
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks_per_chunk = fpga_chunk_size / fpga_column_size;

  // Check if index is within the bounds
  if (idx < fpga_chunk_size) {
    int block_num = idx / fpga_column_size;
    int offset_in_block = idx % fpga_column_size;

    int dest_idx = block_num * fpga_column_size + offset_in_block;

    // Copy from c1, c2, c3, c4 in cyclical manner to c
    c[dest_idx] = c1[idx];
    c[dest_idx + fpga_column_size] = c2[idx];
    c[dest_idx + 2 * fpga_column_size] = c3[idx];
    c[dest_idx + 3 * fpga_column_size] = c4[idx];
    // c[dest_idx] = c1[idx];
  }
}

__global__ void redistribute_8CH(float *c1,
                                 float *c2,
                                 float *c3,
                                 float *c4,
                                 float *c5,
                                 float *c6,
                                 float *c7,
                                 float *c8,
                                 float *c,
                                 int fpga_chunk_size,
                                 int fpga_column_size) {
  // asm volatile ("buffer_wbinvl1_vol");
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks_per_chunk = fpga_chunk_size / fpga_column_size;

  // Check if index is within the bounds
  if (idx < fpga_chunk_size) {
    int block_num = idx / fpga_column_size;
    int offset_in_block = idx % fpga_column_size;

    int dest_idx = block_num * fpga_column_size + offset_in_block;

    // Copy from c1, c2, c3, c4 in cyclical manner to c
    c[dest_idx] = c1[idx];
    c[dest_idx + fpga_column_size] = c2[idx];
    c[dest_idx + 2 * fpga_column_size] = c3[idx];
    c[dest_idx + 3 * fpga_column_size] = c4[idx];
    c[dest_idx + 4 * fpga_column_size] = c5[idx];
    c[dest_idx + 5 * fpga_column_size] = c6[idx];
    c[dest_idx + 6 * fpga_column_size] = c7[idx];
    c[dest_idx + 7 * fpga_column_size] = c8[idx];
    // c[dest_idx] = c1[idx];
  }
}

void check_error(void) {
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
    exit(err);
  }
}

vector<data_type> GPUHandler::transferRedistributeAndDenseMultiply(
    vector<vector<float, aligned_allocator<float>>> &mat_C_fpga_vec, int mat_C_fpga_chunk_size,
    int mat_C_fpga_column_size, int num_ch_c, std::vector<data_type> &l1_weight_matrix, int num_nodes,
    int feature_size, int l1_size) {
  rocblas_status rstatus = rocblas_status_success;

  data_type halpha = static_cast<data_type>(1);
  data_type hbeta = static_cast<data_type>(0);

  const rocblas_operation transA = rocblas_operation_none;
  const rocblas_operation transB = rocblas_operation_none;

  int lda, ldb, ldc, sizeA, sizeB, sizeC;
  int strideA1, strideA2, strideB1, strideB2;

  if (transA == rocblas_operation_none) {
    lda = num_nodes;
    sizeA = feature_size * lda;
    strideA1 = 1;
    strideA2 = lda;
  }
  else {
    lda = feature_size;
    sizeA = num_nodes * lda;
    strideA1 = lda;
    strideA2 = 1;
  }
  if (transB == rocblas_operation_none) {
    ldb = feature_size;
    sizeB = l1_size * ldb;
    strideB1 = 1;
    strideB2 = ldb;
  } else {
    ldb = l1_size;
    sizeB = feature_size * ldb;
    strideB1 = ldb;
    strideB2 = 1;
  }
  ldc = num_nodes;
  sizeC = l1_size * ldc;

  sizeA = mat_C_fpga_chunk_size * 4;

  // using rocblas API
  rocblas_handle handle;
  rstatus = rocblas_create_handle(&handle);
  CHECK_ROCBLAS_STATUS(rstatus);

  // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
  std::vector<data_type> hC_(sizeC);
  std::vector<data_type> hGold(sizeC);

  hGold = hC_;
  {
    // allocate memory on device
    helpers::DeviceVector<data_type> dgemmA_(sizeA);
    helpers::DeviceVector<data_type> dgemmB_(sizeB);
    helpers::DeviceVector<data_type> dgemmC_(sizeC);
    if (num_ch_c == 4) {
      helpers::DeviceVector<data_type> dC1(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC2(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC3(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC4(mat_C_fpga_chunk_size);

      gemmA_moving_time = utils_time_us();

      CHECK_HIP_ERROR(hipMemcpy(
          dC1,
          static_cast<void *>(mat_C_fpga_vec[0].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC2,
          static_cast<void *>(mat_C_fpga_vec[1].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC3,
          static_cast<void *>(mat_C_fpga_vec[2].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC4,
          static_cast<void *>(mat_C_fpga_vec[3].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      HIP_CHECK(hipDeviceSynchronize());
      gemmA_moving_time = (utils_time_us() - gemmA_moving_time) / 1e3;
      redistribute_time = utils_time_us();

      int threadsPerBlock = 1024; // 256;
      int blocksPerGrid = (mat_C_fpga_chunk_size + threadsPerBlock - 1) / threadsPerBlock;

      hipLaunchKernelGGL(redistribute, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, dC1, dC2, dC3, dC4, dgemmA_,
        mat_C_fpga_chunk_size, mat_C_fpga_column_size);

      hipDeviceSynchronize();

      redistribute_time = (utils_time_us() - redistribute_time) / 1e3;
    } else if (num_ch_c == 8) {
      // Define 8 device vectors
      helpers::DeviceVector<data_type> dC1(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC2(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC3(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC4(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC5(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC6(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC7(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC8(mat_C_fpga_chunk_size);

      gemmA_moving_time = utils_time_us();

      // Perform memory copy for each device vector
      CHECK_HIP_ERROR(hipMemcpy(
          dC1,
          static_cast<void *>(mat_C_fpga_vec[0].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC2,
          static_cast<void *>(mat_C_fpga_vec[1].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC3,
          static_cast<void *>(mat_C_fpga_vec[2].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC4,
          static_cast<void *>(mat_C_fpga_vec[3].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC5,
          static_cast<void *>(mat_C_fpga_vec[4].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC6,
          static_cast<void *>(mat_C_fpga_vec[5].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC7,
          static_cast<void *>(mat_C_fpga_vec[6].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC8,
          static_cast<void *>(mat_C_fpga_vec[7].data()),
          sizeof(data_type) * mat_C_fpga_chunk_size,
          hipMemcpyHostToDevice)
      );

      HIP_CHECK(hipDeviceSynchronize());
      gemmA_moving_time = (utils_time_us() - gemmA_moving_time) / 1e3;
      redistribute_time = utils_time_us();

      int threadsPerBlock = 1024; // 256;
      int blocksPerGrid = (mat_C_fpga_chunk_size + threadsPerBlock - 1) / threadsPerBlock;

      // Call the kernel with 8 device vectors
      hipLaunchKernelGGL(redistribute_8CH, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                         dC1, dC2, dC3, dC4, dC5, dC6, dC7, dC8, dgemmA_, mat_C_fpga_chunk_size, mat_C_fpga_column_size);

      HIP_CHECK(hipDeviceSynchronize());
      redistribute_time = (utils_time_us() - redistribute_time) / 1e3;
    } else {
      std::cout << "num_ch_c:" << num_ch_c <<" NUM_CH_C should be 4 or 8" <<
      endl;
      exit(0);
    }

    gemmBC_move_compute_and_C_move_time = utils_time_us();
    CHECK_HIP_ERROR(hipMemcpy(
        dgemmB_, static_cast<void *>(l1_weight_matrix.data()), sizeof(data_type) * sizeB, hipMemcpyHostToDevice)
    );
    CHECK_HIP_ERROR(hipMemcpy(
        dgemmC_, static_cast<void *>(hC_.data()), sizeof(data_type) * sizeC, hipMemcpyHostToDevice)
    );

    // enable passing alpha parameter from pointer to host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    // asynchronous calculation on device, returns before finished calculations
    rstatus = rocblas_sgemm(
        handle,
        transA,
        transB,
        num_nodes,
        l1_size,
        feature_size,
        &halpha,
        dgemmA_,
        lda,
        dgemmB_,
        ldb,
        &hbeta,
        dgemmC_,
        ldc);

    // check that calculation was launched correctly on device, not that result
    // was computed yet
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch device memory results, automatically blocked until results ready
    CHECK_HIP_ERROR(hipMemcpy(hC_.data(), dgemmC_, sizeof(data_type) * sizeC, hipMemcpyDeviceToHost)
    );

    HIP_CHECK(hipDeviceSynchronize());
    gemmBC_move_compute_and_C_move_time = (utils_time_us() - gemmBC_move_compute_and_C_move_time) / 1e3;
  } // release device memory via helpers::DeviceVector destructors

  rstatus = rocblas_destroy_handle(handle);
  CHECK_ROCBLAS_STATUS(rstatus);
  return hC_;
}

vector<data_type> GPUHandler::transferRedistributeAndDenseMultiply(
    vector<float*> &mat_C_hostmap, int mat_C_fpga_chunk_size, int mat_C_fpga_column_size, int num_ch_c,
    std::vector<data_type> &l1_weight_matrix, int num_nodes, int feature_size, int l1_size) {
  rocblas_status rstatus = rocblas_status_success;

  data_type halpha = static_cast<data_type>(1);
  data_type hbeta = static_cast<data_type>(0);

  const rocblas_operation transA = rocblas_operation_none;
  const rocblas_operation transB = rocblas_operation_none;

  int lda, ldb, ldc, sizeA, sizeB, sizeC;
  int strideA1, strideA2, strideB1, strideB2;

  if (transA == rocblas_operation_none) {
    lda = num_nodes;
    sizeA = feature_size * lda;
    strideA1 = 1;
    strideA2 = lda;
  } else {
    lda = feature_size;
    sizeA = num_nodes * lda;
    strideA1 = lda;
    strideA2 = 1;
  }
  if (transB == rocblas_operation_none) {
    ldb = feature_size;
    sizeB = l1_size * ldb;
    strideB1 = 1;
    strideB2 = ldb;
  } else {
    ldb = l1_size;
    sizeB = feature_size * ldb;
    strideB1 = ldb;
    strideB2 = 1;
  }
  ldc = num_nodes;
  sizeC = l1_size * ldc;

  sizeA = mat_C_fpga_chunk_size * 4;

  // using rocblas API
  rocblas_handle handle;
  rstatus = rocblas_create_handle(&handle);
  CHECK_ROCBLAS_STATUS(rstatus);

  // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
  std::vector<data_type> hC_(sizeC);
  std::vector<data_type> hGold(sizeC);

  hGold = hC_;
  {
    // allocate memory on device
    helpers::DeviceVector<data_type> dgemmA_(sizeA);
    helpers::DeviceVector<data_type> dgemmB_(sizeB);
    helpers::DeviceVector<data_type> dgemmC_(sizeC);

    if (num_ch_c == 4) {

      helpers::DeviceVector<data_type> dC1(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC2(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC3(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC4(mat_C_fpga_chunk_size);

      float *dC_ptr[4];

      for (int i = 0; i < 4; i++) {
        hipHostRegister((void *)mat_C_hostmap[i], sizeof(float) * (mat_C_fpga_chunk_size), hipHostRegisterMapped);
        check_error();
        hipHostGetDevicePointer((void **)&dC_ptr[i], (void *)mat_C_hostmap[i], 0);
        check_error();
      }
      gemmA_moving_time = utils_time_us();

      CHECK_HIP_ERROR(hipMemcpy(
          dC1, static_cast<void *>(dC_ptr[0]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC2, static_cast<void *>(dC_ptr[1]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC3, static_cast<void *>(dC_ptr[2]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC4, static_cast<void *>(dC_ptr[3]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );
      HIP_CHECK(hipDeviceSynchronize());
      gemmA_moving_time = (utils_time_us() - gemmA_moving_time) / 1e3;

      int threadsPerBlock = 1024;
      int blocksPerGrid = (mat_C_fpga_chunk_size + threadsPerBlock - 1) / threadsPerBlock;

      redistribute_time = utils_time_us();
      hipLaunchKernelGGL(redistribute, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, dC1, dC2, dC3, dC4, dgemmA_,
        mat_C_fpga_chunk_size, mat_C_fpga_column_size);

      hipDeviceSynchronize();

      redistribute_time = (utils_time_us() - redistribute_time) / 1e3;
    } else if (num_ch_c == 8) {

      helpers::DeviceVector<data_type> dC1(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC2(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC3(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC4(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC5(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC6(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC7(mat_C_fpga_chunk_size);
      helpers::DeviceVector<data_type> dC8(mat_C_fpga_chunk_size);

      float *dC_ptr[8];

      for (int i = 0; i < 8; i++) {
        hipHostRegister((void *)mat_C_hostmap[i], sizeof(float) * (mat_C_fpga_chunk_size), hipHostRegisterMapped);
        check_error();
        hipHostGetDevicePointer((void **)&dC_ptr[i], (void *)mat_C_hostmap[i], 0);
        check_error();
      }
      gemmA_moving_time = utils_time_us();

      CHECK_HIP_ERROR(hipMemcpy(
          dC1, static_cast<void *>(dC_ptr[0]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC2, static_cast<void *>(dC_ptr[1]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC3, static_cast<void *>(dC_ptr[2]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC4, static_cast<void *>(dC_ptr[3]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC1, static_cast<void *>(dC_ptr[4]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC2, static_cast<void *>(dC_ptr[5]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC3, static_cast<void *>(dC_ptr[6]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );

      CHECK_HIP_ERROR(hipMemcpy(
          dC4, static_cast<void *>(dC_ptr[7]), sizeof(data_type) * mat_C_fpga_chunk_size, hipMemcpyHostToDevice)
      );
      HIP_CHECK(hipDeviceSynchronize());
      gemmA_moving_time = (utils_time_us() - gemmA_moving_time) / 1e3;

      int threadsPerBlock = 1024;
      int blocksPerGrid = (mat_C_fpga_chunk_size + threadsPerBlock - 1) / threadsPerBlock;

      redistribute_time = utils_time_us();
      hipLaunchKernelGGL(redistribute_8CH, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
        dC1, dC2, dC3, dC4, dC5, dC6, dC7, dC8, dgemmA_,
      mat_C_fpga_chunk_size, mat_C_fpga_column_size);

      hipDeviceSynchronize();

      redistribute_time = (utils_time_us() - redistribute_time) / 1e3;
    } else {
      std::cout << "num_ch_c:" << num_ch_c <<" NUM_CH_C should be 4 or 8" <<
      endl;
      exit(0);
    }

    gemmBC_move_compute_and_C_move_time = utils_time_us();
    CHECK_HIP_ERROR(hipMemcpy(
        dgemmB_, static_cast<void *>(l1_weight_matrix.data()), sizeof(data_type) * sizeB, hipMemcpyHostToDevice)
    );
    CHECK_HIP_ERROR(hipMemcpy(
        dgemmC_, static_cast<void *>(hC_.data()), sizeof(data_type) * sizeC, hipMemcpyHostToDevice)
    );

    // enable passing alpha parameter from pointer to host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    // asynchronous calculation on device, returns before finished calculations
    rstatus = rocblas_sgemm(
        handle,
        transA,
        transB,
        num_nodes,
        l1_size,
        feature_size,
        &halpha,
        dgemmA_,
        lda,
        dgemmB_,
        ldb,
        &hbeta,
        dgemmC_,
        ldc);

    // check that calculation was launched correctly on device, not that result
    // was computed yet
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch device memory results, automatically blocked until results ready
    CHECK_HIP_ERROR(hipMemcpy(hC_.data(), dgemmC_, sizeof(data_type) * sizeC, hipMemcpyDeviceToHost)
    );

    HIP_CHECK(hipDeviceSynchronize());
    gemmBC_move_compute_and_C_move_time = (utils_time_us() - gemmBC_move_compute_and_C_move_time) / 1e3;
  } // release device memory via helpers::DeviceVector destructors

  rstatus = rocblas_destroy_handle(handle);
  CHECK_ROCBLAS_STATUS(rstatus);
  return hC_;
}

void GPUHandler::displayPerformanceMetrics() {

  // std::cout << std::fixed << std::setprecision(3);

  std::cout << "==============================================" << std::endl;
  std::cout << "       GPU PERFORMANCE METRICS" << std::endl;
  std::cout << "==============================================" << std::endl
            << std::endl;

  std::cout << "------ SETUP TIMINGS ------" << std::endl;
  std::cout << "A - Move Time:                     " << spmmA_moving_time_cpu << " msec" << std::endl;
  std::cout << "Cin - Move Time:                   " << spmmCin_moving_time_cpu << " msec" << std::endl
            << std::endl;

  std::cout << "------ GPU SPDMM TIMINGS (CPU PERSPECTIVE) ------" << std::endl;
  std::cout << "GPU SPDMM:                         " << gpu_spdmm_total_time << " msec" << std::endl
            << std::endl;

  std::cout << "Dense B Move Time:                 " << spmmB_moving_time_cpu << " msec" << std::endl;
  std::cout << "Kernel Execution Time:             " << spmm_kernel_exec_time << " msec" << std::endl;
  std::cout << "Output Buffer Move:                " << spmmCout_moving_time_cpu << " msec" << std::endl
            << std::endl;

  std::cout << "------ DATA THROUGHPUT & SIZE ------" << std::endl;
  std::cout << "Cin Throughput:                    " << cinThroughput << " GB/s      | Size: " << cinSize << " MB"
            << std::endl;
  std::cout << "Cout Throughput (g2c):             " << coutThroughput << " GB/s" << std::endl
            << std::endl;

  std::cout << "------ GPU DeMM TIMINGS (CPU PERSPECTIVE) ------" << std::endl;
  std::cout << "GPU DeMM:                          " << gpu_demm_total_time << " msec" << std::endl
            << std::endl;

  std::cout << "A Move Time:                       " << gemmA_moving_time << " msec" << std::endl;
  std::cout << "Redistribute Time:                 " << redistribute_time << " msec" << std::endl;
  std::cout << "B C move and Kernel Execution Time:" << gemmBC_move_compute_and_C_move_time << " msec" << std::endl
            << std::endl;
}

void GPUHandler::initializePerformanceMetrics() {
  spmmB_moving_time_cpu = 0;
  spmmCin_moving_time_cpu = 0;
  spmmA_moving_time_cpu = 0;
  spmmCout_moving_time_cpu = 0;

  gemmA_moving_time = 0;
  redistribute_time = 0;
  gemmBC_move_compute_and_C_move_time = 0;

  cinThroughput = 0;
  coutThroughput = 0;
  cinSize = 0;
}

// Return spdmm total time, B prepare time, B move time, Execution time, Output move time
std::string GPUHandler::csvUpdateSPMMTime() {
  std::ostringstream ss;
  ss << gpu_spdmm_total_time << ","
     << "0"
     << ","
     << spmmB_moving_time_cpu << ","
     << spmm_kernel_exec_time << ","
     << spmmCout_moving_time_cpu;
  return ss.str();
}

// Return gemm total time, A move time, Destribute time, BC move+exec+C move time
std::string GPUHandler::csvUpdateGEMMTime() {
  std::ostringstream ss;
  ss << gpu_demm_total_time << ","
     << gemmA_moving_time << ","
     << redistribute_time << ","
     << gemmBC_move_compute_and_C_move_time;
  return ss.str();
}

// Return
std::string GPUHandler::csvUpdateSetupTime() {
  std::ostringstream ss;
  ss << "0"
     << ","
     << "0"
     << ","
     << spmmA_moving_time_cpu << ","
     << spmmCin_moving_time_cpu;
  return ss.str();
}
