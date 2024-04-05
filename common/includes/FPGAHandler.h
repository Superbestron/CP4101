#ifndef FPGA_HANDLER_HPP
#define FPGA_HANDLER_HPP

#include <vector>
#include <ap_int.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "xcl2/xcl2.h"

#include "mmio.h"
#include "sparse_helper.h"
#ifdef __HIPCC__
// #define HIP_FPGA_P2P
// #define HIP_FPGA_NO_P2P
#endif
#ifndef __HIPCC__
#include <torch/torch.h>
#endif

class FPGAHandler {
 private:
  static const int NUM_CH_SPARSE = 4; // 8;
  static const int NUM_CH_B = 4;
  static const int NUM_CH_C = 4; // 8;
  static const int NUM_WINDOW_SIZE = 4096;
  static const int rp_time = 1;

  cl::Context context;
  cl::Program program;
  cl::CommandQueue q;
  cl::Kernel krnl_sextans;
  cl_int err;

  vector<vector<edge>> edge_list_pes;
  vector<unsigned int> edge_list_ptr;
  vector<unsigned int, aligned_allocator<unsigned int>> edge_list_ptr_fpga;

  vector<vector<unsigned long, aligned_allocator<unsigned long>>>
  sparse_A_fpga_vec;

  vector<vector<float, aligned_allocator<float>>>
  mat_B_fpga_vec;
  int mat_B_fpga_column_size;
  int mat_B_fpga_chunk_size;

  vector<vector<float, aligned_allocator<float>>>
  mat_C_fpga_in;

  int mat_C_fpga_in_column_size;
  int mat_C_fpga_in_chunk_size;

  std::vector<cl::Buffer> buffer_A;
  std::vector<cl::Buffer> buffer_B;
  std::vector<cl::Buffer> buffer_C_in;
  std::vector<cl::Buffer> buffer_C;
  std::vector<cl::Buffer> p2pbuffer_C;

  double preparing_A_time_cpu = 0;
  double preparing_B_time_cpu = 0;
  double preparing_C_time_cpu = 0;
  double buffer_creation_time_cpu = 0;
  double kernel_creation_time = 0;
  double B_moving_time_cpu = 0;
  double kernel_launch_time = 0;
  double Cout_moving_time_cpu = 0;
  double f2g_p2p_datamovement_time = 0;

  float B_partition_moving_time_ocl = 0;
  float A_partition_moving_time_ocl = 0;
  float Cin_partition_moving_time_ocl = 0;
  float Cout_partition_moving_time_ocl = 0;
  float kernel_time_ocl = 0;

  float B_moving_time_ocl = 0;
  float A_moving_time_ocl = 0;
  float Cin_moving_time_ocl = 0;
  float Cout_moving_time_ocl = 0;

  double cinThroughput = 0;
  double coutThroughput = 0;

  cl::Event A_migration_events[NUM_CH_SPARSE];
  cl::Event B_migration_events[NUM_CH_B];
  cl::Event Cin_migration_events[NUM_CH_C];
  cl::Event Cout_migration_events[NUM_CH_C];
  cl::Event Ptr_migration_event;
  cl::Event kernel_launch_event;

  std::string m_bitstreamPath;
  double utils_time_us(void);
  void check_error(void);
  void computeEventsTime(float &average_time_per_event,
                         float &start_to_end_time,
                         cl::Event migration_events[],
                         const std::string &name,
                         int num_events);

 public:
  FPGAHandler() : sparse_A_fpga_vec(NUM_CH_SPARSE),
                  mat_B_fpga_vec(NUM_CH_B),
                  mat_C_fpga_in(NUM_CH_C),
                  spmm_output_C_vec(NUM_CH_C),
                  spmm_output_C_hostmap(NUM_CH_C),
                  num_ch_c(NUM_CH_C) {};

  // ~FPGAHandler(){};

  void LoadBitstream(
      const std::string &binaryFile,
      const std::string &targetDeviceName);

  void setupData(
      const SparseMatrix &sparse_matrix,
      vector<float> &mat_C_cpu,
      const int &N,
      const float &ALPHA,
      const float &BETA,
      const bool enableP2P);

  void setupP2PBuffers();

  void executeSparseMultiplication(
      const SparseMatrix &sparse_matrix,
      const vector<float> &mat_B_cpu,
      const int &N,
      const float &ALPHA,
      const float &BETA);

  void verifyFPGAResults(vector<float> &mat_C_cpu, int M, int N);

  void initializePerformanceMetrics();
  void displayPerformanceMetrics();

  std::string csvUpdateSPMMTime();
  std::string csvUpdateSetupTime();
  std::string csvUpdateAllTime();

  void mapFPGABufferToHost();

  void transferOutputToCPU();

  double calculateAveragePower(const std::string &filename);

  double fpga_spdmm_sextans_time = 0;
  double fpgaExecutionTime = 0;
  double fpgaGFLOPS = 0;
  double cpuExecutionTime = 0;
  double cpuGFLOPS = 0;

  vector<vector<float, aligned_allocator<float>>>
  spmm_output_C_vec;
  int mat_C_fpga_column_size;
  int mat_C_fpga_chunk_size;
  int num_ch_c;
  vector<float*> spmm_output_C_hostmap;
};

#endif // FPGA_HANDLER_HPP
