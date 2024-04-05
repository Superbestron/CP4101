#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define DEBUG
// #define PRINT_MATRIX
// #define ITER 1
#include <vector>
#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <climits>
#include <sys/stat.h>
#include <cstdio>
#include <ap_int.h>
#include <cstdlib>
#include <ctime>
#include <iterator>
#include <string>
#include <cfloat>
#include <random>
#include <filesystem>

#include "../common/includes/GPUHandler.h"

using namespace std;

string matrix_details_str;
char *filename_A;
int ITER;

inline int ceil_eightx(int x) {
  if (x <= 0)
    return 1;
  return ((x + 7) / 8) * 8;
}

inline double utils_time_us(void) {
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  return (static_cast<double>(duration));
};

void initialize_matrices(char *filename_A,
                         SparseMatrix &sparse_matrix,
                         vector<float> &mat_B_cpu,
                         vector<float> &mat_C_cpu,
                         int &N) {

  cout << "Reading sparse A matrix...";
  double read_suitsparse_matrix_time = utils_time_us();
  read_suitsparse_matrix(filename_A,
                         sparse_matrix.CSRRowPtr,
                         sparse_matrix.CSRColIndex,
                         sparse_matrix.CSRVal,
                         sparse_matrix.M,
                         sparse_matrix.K,
                         sparse_matrix.nnz,
                         CSR);

  // for (int i = 0; i < 10; i++) cout << sparse_matrix.CSRRowPtr[i] << '\n';
  // cout << '\n';
  // for (int i = 0; i < 10; i++) cout << sparse_matrix.CSRColIndex[i] << '\n';
  // cout << '\n';
  // for (int i = 0; i < 10; i++) cout << sparse_matrix.CSRVal[i] << '\n';
  // cout << '\n';

  // read_suitsparse_matrix(filename_A,
  //                        sparse_matrix.CSCColPtr,
  //                        sparse_matrix.CSCRowIndex,
  //                        sparse_matrix.CSCVal,
  //                        sparse_matrix.M,
  //                        sparse_matrix.K,
  //                        sparse_matrix.nnz,
  //                        CSC);
  read_suitsparse_matrix_time = (utils_time_us() - read_suitsparse_matrix_time) / (1e3);
  std::cout << "read_suitsparse_matrix_time (msec):   " << read_suitsparse_matrix_time << std::endl;

  cout << "done\n";
  int M = sparse_matrix.M;
  int K = sparse_matrix.K;
  int nnz = sparse_matrix.nnz;

  cout << "Matrix size: \n";
  cout << "A: sparse matrix, " << M << " x " << K << ". NNZ = " << nnz << "\n";
  cout << "B: dense matrix, " << K << " x " << N << "\n";
  cout << "C: dense matrix, " << M << " x " << N << "\n";

  // initiate matrix B and matrix C
  mat_B_cpu.resize(K * N, 0.0);
  mat_C_cpu.resize(M * N, 0.0);

  cout << "Generating dense matrix B ...";
  for (int nn = 0; nn < N; ++nn) {
    for (int kk = 0; kk < K; ++kk) {
      if (nn == kk)
        mat_B_cpu[kk + K * nn] = 1; //(1.0 + kk) + 0.1 * (1.0 + nn); //100.0 * (kk + 1)  + 1.0 * (nn + 1);// / K / N;
    }
  }

  cout << "Generating dense matrix C ...";
  for (int nn = 0; nn < N; ++nn) {
    for (int mm = 0; mm < M; ++mm) {
      mat_C_cpu[mm + M * nn] = 1.0 * (mm + 1) * (nn + 1);
    }
  }

  cout << "done\n";
}

void read_file(std::string filePath, std::vector<data_type> &data_array) {
  std::ifstream ifd(filePath, std::ios::binary | std::ios::ate);
  int data_size = ifd.tellg();
  ifd.seekg(0, std::ios::beg);
  if (ifd.fail()) {
    std::cout << "ERROR! " << filePath << " is not opened correctly." << std::endl;
  }

  std::vector<data_type> buffer;
  buffer.resize(data_size / sizeof(data_type));
  ifd.read((char *) buffer.data(), data_size);
  for (float i : buffer)
    data_array.push_back(i);
}

void updateCSV(const std::string &filename,
               string s1,
               string s2,
               string s3,
               string s4,
               string s5,
               string s6,
               string s7,
               string s8,
               string s9) {
  std::ofstream file;

  // Check if file exists
  bool fileExists = std::filesystem::exists(filename);

  file.open(filename, std::ios_base::app | std::ios_base::out); // append mode

  // If the file doesn't exist, write the headers first
  if (!fileExists) {
    file
        << "mat name, M, K, N, l1_size, sparsity(%), SPMM, denseMM, Layer,total_time, spmm_time, gemm_time, spdmm, prepB, B_move, exec, out_move, gemm, A_move, redistribute, BC_move_exec\n";
    // Add all other metric headers here
  }

  // Now write the data
  file << s1 << ","
       << s2 << ","
       << s3 << ","
       << s4 << ","
       << s5 << ","
       << s6 << ","
       << s7 << ","
       << s8 << ","
       << s9 << ","
       << "\n";

  file.close();
}

void method3(GPUHandler &gpu_instance,
             SparseMatrix &sparse_matrix,
             vector<float> &mat_B_cpu,
             int N,
             int ALPHA,
             int BETA,
             int num_nodes,
             int feature_size,
             int iter) {
  std::srand(std::time(nullptr));
  gpu_instance.initializePerformanceMetrics();

  gpu_instance.gpu_spdmm_total_time = utils_time_us();
  gpu_instance.executeSparseMultiplication(sparse_matrix, mat_B_cpu, N, ALPHA, BETA);
  gpu_instance.gpu_spdmm_total_time = (utils_time_us() - gpu_instance.gpu_spdmm_total_time) / (1e3);

  if (iter == ITER - 1) {
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "   METHOD3: Layer 1: GPU SPMM -> GPU denseMM" << std::endl;
    std::cout << "==============================================" << std::endl
              << std::endl;
    gpu_instance.displayPerformanceMetrics();

    std::ostringstream ss;
    ss << "" << ","
       << sparse_matrix.M << ","
       << sparse_matrix.K << ","
       << N << ","
       << sparse_matrix.sparsity;
    matrix_details_str = ss.str();
    std::cout << std::to_string(gpu_instance.gpu_spdmm_total_time) << gpu_instance.csvUpdateSPMMTime()
              << gpu_instance.csvUpdateSetupTime() << std::endl;
  }
}

int main(int argc, char **argv) {
  printf("start host\n");

  srand(0);

  float ALPHA = 1; // 1;//0.85;
  float BETA = 0;  // 1;//-2.06;

  int N = ceil_eightx(stoi(argv[2]));//300;
  filename_A = argv[1];
  ITER = stoi(argv[3]);

  cout << "N = " << N << "\n";
  cout << "alpha = " << ALPHA << "\n";
  cout << "beta = " << BETA << "\n";

  SparseMatrix sparse_matrix;

  vector<float> mat_B_cpu, mat_C_cpu;

  initialize_matrices(filename_A, sparse_matrix, mat_B_cpu, mat_C_cpu, N);

  int num_nodes = sparse_matrix.M;
  int feature_size = N;

  double sparse_matrix_size = static_cast<double>(sparse_matrix.M) * static_cast<double>(sparse_matrix.K);
  double sparsity = ((sparse_matrix_size - static_cast<double>(sparse_matrix.nnz)) / sparse_matrix_size) * 100;
  sparse_matrix.sparsity = sparsity;

  double time_data_read = utils_time_us();

  time_data_read = (utils_time_us() - time_data_read) / (1e3);
  std::cout << "data read time (msec):   " << time_data_read << std::endl;

  GPUHandler gpu_instance;

  gpu_instance.setupData(sparse_matrix, mat_B_cpu, N, ALPHA, BETA);

#if 0
  verifyFPGAResults(fpga_instance, sparse_matrix, mat_C_cpu, mat_B_cpu, N, ALPHA, BETA);
    return 0;
#endif

  int iter = ITER; // Total number of iterations for each method execution

  // Method 3:
  // Layer 1: GPU SPMM -> GPU denseMM
  // Layer 2: GPU SPMM -> GPU denseMM
  double method3_time = utils_time_us();
  for (int i = 0; i < iter; i++) {
    method3(gpu_instance, sparse_matrix, mat_B_cpu, N, ALPHA, BETA, num_nodes, feature_size, i);
  }
  method3_time = (utils_time_us() - method3_time) / iter / (1e3);
  // add an empty line
  updateCSV("result.csv", "", "", "", "", "", "", "", "", "");

  // std::cout << "------ METHOD TIMING and POWER------" << std::endl;
  // std::cout << "Method 1: FPGA SPMM -> CPU -> GPU denseMM:  TIME:" << method1_time << " msec, FPGA Power:" << method1_fpga_power<< std::endl;
  // std::cout << "Method 2: FPGA SPMM -> GPU denseMM:         TIME:" << method2_time << " msec, FPGA Power:" << method2_fpga_power<< std::endl;
  // std::cout << "Method 3: GPU SPMM -> CPU -> GPU denseMM:   TIME:" << method3_time << " msec" << std::endl;

  return 0;

}
