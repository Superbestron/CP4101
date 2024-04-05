#ifndef GPU_HANDLER_HPP
#define GPU_HANDLER_HPP

#include <vector>
#include <ap_int.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "xcl2/xcl2.h"

#include "sparse_helper.h"

typedef float data_type;
typedef int32_t index_type;

class GPUHandler {
 private:
  index_type *dAptr = NULL;
  index_type *dAcol = NULL;
  data_type *dAval = NULL;
  data_type *dB = NULL;
  data_type *dspmmC = NULL;

  std::vector<data_type> hC;

  double spmmB_moving_time_cpu = 0;
  double spmmCin_moving_time_cpu = 0;
  double spmmA_moving_time_cpu = 0;
  double spmmCout_moving_time_cpu = 0;

  double gemmA_moving_time = 0;
  double redistribute_time = 0;
  double gemmBC_move_compute_and_C_move_time = 0;

  double cinThroughput = 0;
  double coutThroughput = 0;
  double cinSize = 0;

  double utils_time_us(void);

 public:
  GPUHandler() {

  };

  void setupData(
      const SparseMatrix &sparse_matrix,
      vector<float> &mat_C_cpu,
      const int &N,
      const float &ALPHA,
      const float &BETA);

  void executeSparseMultiplication(
      const SparseMatrix &sparse_matrix,
      const vector<float> &feature_matrix,
      const int &N,
      const float &ALPHA,
      const float &BETA);

  vector<data_type> executeDenseMultiplication(
      std::vector<data_type> &l1_weight_matrix,
      int num_nodes,
      int feature_size,
      int l1_size);

  void verifyGPUResults(vector<float> &mat_C_cpu, int M, int N);

  void displayPerformanceMetrics();
  void initializePerformanceMetrics();

  std::string csvUpdateSPMMTime();
  std::string csvUpdateGEMMTime();
  std::string csvUpdateSetupTime();

#ifdef HIP_FPGA_P2P
  void performFpgaToGpuTransfer();
#endif

  vector<data_type> transferRedistributeAndDenseMultiply(
      vector<vector<float, aligned_allocator<float>>> &mat_C_fpga_vec,
    int mat_C_fpga_chunk_size,
    int mat_C_fpga_column_size,
    int num_ch_c,
    std::vector<data_type> &l1_weight_matrix,
    int num_nodes,
    int feature_size,
    int l1_size
  );

  vector<data_type> transferRedistributeAndDenseMultiply(
      vector<float *> &mat_C_hostmap,
      int mat_C_fpga_chunk_size,
      int mat_C_fpga_column_size,
      int num_ch_c,
      std::vector<data_type> &l1_weight_matrix,
      int num_nodes,
      int feature_size,
      int l1_size);

  double gpu_spdmm_total_time = 0;
  double gpu_demm_total_time = 0;
  double spmm_kernel_exec_time = 0;
  double gpuGFLOPS = 0;
  double cpuExecutionTime = 0;
  double cpuGFLOPS = 0;
};

#endif // GPU_HANDLER_HPP
