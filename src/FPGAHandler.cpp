#include "../common/includes/FPGAHandler.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <sstream>
#define DEBUG

void FPGAHandler::LoadBitstream(const std::string &binaryFile, const std::string &targetDeviceName) {
  std::cout << "Loading bitstream from: " << m_bitstreamPath << std::endl;
  // Logic to load the FPGA bitstream
  cl_int err;
  // unsigned fileBufSize;

  // OPENCL HOST CODE AREA START
  auto devices = xcl::get_xil_devices();
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

  for (size_t i = 0; i < devices.size(); i++) {
    std::cout << "Device name " << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
  }
  cl::Device device;

  for (size_t i = 0; i < devices.size(); i++) {
    std::string deviceName = devices[i].getInfo<CL_DEVICE_NAME>();
    // std::cout << "Device " << i << ": " << deviceName << std::endl;

    if (deviceName == targetDeviceName) {
      std::cerr << "Target device " << targetDeviceName << " found." << std::endl;
      device = devices[i];
      break;
    }
    if (i == devices.size() - 1) {
      std::cerr << "Target device " << targetDeviceName << " not found." << std::endl;
      return;
      // std::tuple<cl::Context, cl::Program, cl::CommandQueue>(); // return an empty tuple
    }
  }

  OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err,
            q = cl::CommandQueue(context,
                                 device,
                                 CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                 &err));
  OCL_CHECK(err, std::string
  device_name = device.getInfo<CL_DEVICE_NAME>(&err));

#ifdef DEBUG
  std::cout << "FPGA binary read..." << std::endl;
#endif

  // cl::Program::Binaries bins{{fileBuf, fileBufSize}};
  devices.resize(1);
  OCL_CHECK(err, program = cl::Program(context, {device}, bins, NULL, &err));

  OCL_CHECK(err, krnl_sextans = cl::Kernel(program, "sextans", &err));
}

void FPGAHandler::computeEventsTime(float &average_time_per_event,
                                    float &start_to_end_time,
                                    cl::Event migration_events[],
                                    const std::string &name,
                                    int num_events) {
  uint64_t nstimestart, nstimeend;
  cl_int err;
  average_time_per_event = 0;
  for (int i = 0; i < num_events; i++) {
    OCL_CHECK(err,
              err = migration_events[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err,
              err = migration_events[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    average_time_per_event += nstimeend - nstimestart;

    // cout << "time for each iteration (msec): start:" << nstimestart/1e6 << " end:" << nstimeend/1e6 << " elapsed:" << (nstimeend - nstimestart)/1e6 << endl;
  }

  OCL_CHECK(err,
            err = migration_events[0].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
  OCL_CHECK(err,
            err = migration_events[num_events - 1].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
  start_to_end_time = nstimeend - nstimestart;
  average_time_per_event = average_time_per_event / 1e6 / num_events;
  start_to_end_time = start_to_end_time / 1e6;

  // cout << "=========="<<name<<" event time (msec) ===========" << endl;
  // cout << "average time:" << average_time_per_event<< endl;
  // cout << "elapsed moving time from start of first and end of last:" << start_to_end_time << endl;
  // cout << "====================================" << endl;
}

inline double FPGAHandler::utils_time_us(void) {
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  return (static_cast<double>(duration));
};

void FPGAHandler::setupData(
    const SparseMatrix &sparse_matrix,
    vector<float> &mat_C_cpu,
    const int &N,
    const float &ALPHA,
    const float &BETA,
    const bool enableP2P) {
  // generate for fpga
  cout << "Preparing sparse A for FPGA ...";
  preparing_A_time_cpu = utils_time_us();

  int NUM_PE;
  int WINDOE_SIZE;
  generate_edge_list_for_all_PEs(sparse_matrix.CSCColPtr,       // const vector<int> & CSCColPtr,
                                 sparse_matrix.CSCRowIndex,     // const vector<int> & CSCRowIndex,
                                 sparse_matrix.CSCVal,          // const vector<float> & CSCVal,
                                 NUM_PE = NUM_CH_SPARSE * 8,    // const int NUM_PE,
                                 sparse_matrix.M_CSC,           // const int NUM_ROW,
                                 sparse_matrix.K_CSC,           // const int NUM_COLUMN,
                                 WINDOE_SIZE = NUM_WINDOW_SIZE, // const int WINDOE_SIZE,
                                 edge_list_pes,                 // vector<vector<edge> > & edge_list_pes,
                                 edge_list_ptr,                 // vector<int> & edge_list_ptr,
                                 10);                           // const int DEP_DIST_LOAD_STORE = 10)

  // vector<unsigned int, aligned_allocator<unsigned int> > edge_list_ptr_fpga;
  int edge_list_ptr_fpga_size = ((edge_list_ptr.size() + 15) / 16) * 16;
  int edge_list_ptr_fpga_chunk_size = ((edge_list_ptr_fpga_size + 1023) / 1024) * 1024;
  edge_list_ptr_fpga.resize(edge_list_ptr_fpga_chunk_size, 0);
  for (unsigned int i = 0; i < edge_list_ptr.size(); ++i) {
    edge_list_ptr_fpga[i] = edge_list_ptr[i];
  }

#ifdef DEBUG_PRINT
  cout << "\n ############## DEBUG PRINT ################# \n";
    cout << "edge_list_ptr_fpga_size = " << edge_list_ptr_fpga_size << endl;
    cout << "edge_list_ptr_fpga_chunk_size = " << edge_list_ptr_fpga_chunk_size << endl;
    cout << "edge_list_ptr_fpga  = \n";
    for (unsigned int i = 0; i < edge_list_ptr.size(); ++i)
    {
        cout << edge_list_ptr_fpga[i] << endl;
    }
    cout << endl;
#endif

  // vector<vector<unsigned long, aligned_allocator<unsigned long> > > sparse_A_fpga_vec(NUM_CH_SPARSE);
  int sparse_A_fpga_column_size = 8 * edge_list_ptr[edge_list_ptr.size() - 1] * 4 / 4;
  // int sparse_A_fpga_chunk_size = ((sparse_A_fpga_column_size + 511)/512) * 512;

  edge_list_64bit(edge_list_pes,
                  edge_list_ptr,
                  sparse_A_fpga_vec,
                  NUM_CH_SPARSE);

#ifdef DEBUG_PRINT
  cout << "\n ############## DEBUG PRINT ################# \n";
    cout << "sparse_A_fpga_column_size = " << sparse_A_fpga_column_size << endl;
    // cout << "sparse_A_fpga_chunk_size = " << sparse_A_fpga_chunk_size << endl;
    cout << endl;
#endif

  preparing_A_time_cpu = (utils_time_us() - preparing_A_time_cpu) / (1e3);
  cout << "done\n";

  int M = sparse_matrix.M;
  int K = sparse_matrix.K;
  int nnz = sparse_matrix.nnz;

  if (NUM_CH_B == 8) {
    mat_B_fpga_column_size = ((K + 16 - 1) / 16) * 16;
  } else if (NUM_CH_B == 4) {
    mat_B_fpga_column_size = ((K + 8 - 1) / 8) * 8 * 2;
  }
  mat_B_fpga_chunk_size = ((mat_B_fpga_column_size * (N / NUM_CH_B) + 1023) / 1024) * 1024;

#ifdef DEBUG_PRINT
  cout << "\n ############## DEBUG PRINT ################# \n";
    cout << "mat_B_fpga_column_size = " << mat_B_fpga_column_size << endl;
    cout << "mat_B_fpga_chunk_size = " << mat_B_fpga_chunk_size << endl;
    cout << endl;
#endif

  for (int cc = 0; cc < NUM_CH_B; ++cc) {
    mat_B_fpga_vec[cc].resize(mat_B_fpga_chunk_size, 0.0);
  }
#if 1
  mat_C_fpga_in_column_size = ((M + 16 - 1) / 16) * 16;
  mat_C_fpga_in_chunk_size = ((mat_C_fpga_in_column_size * (N / NUM_CH_C) + 1023) / 1024) * 1024;
#else
  mat_C_fpga_in_column_size = ((M + 8 - 1) / 8) * 8 * 2;
    mat_C_fpga_in_chunk_size = ((mat_C_fpga_in_column_size * (N / 2) + 1023) / 1024) * 1024;
#endif

  for (int nn = 0; nn < NUM_CH_C; ++nn) {
    mat_C_fpga_in[nn].resize(mat_C_fpga_in_chunk_size, 0.0);
  }

#if 1
  mat_C_fpga_column_size = ((M + 16 - 1) / 16) * 16;
  mat_C_fpga_chunk_size = ((mat_C_fpga_column_size * (N / NUM_CH_C) + 1023) / 1024) * 1024;
#else
  mat_C_fpga_column_size = ((M + 8 - 1) / 8) * 8 * 2;
    mat_C_fpga_chunk_size = ((mat_C_fpga_column_size * (N / 2) + 1023) / 1024) * 1024;
#endif

  for (int cc = 0; cc < NUM_CH_C; ++cc) {
    spmm_output_C_vec[cc] = vector < float, aligned_allocator < float >> (mat_C_fpga_chunk_size, 0.0);
  }

  cout << "Preparing dense C for FPGA ...";
  preparing_C_time_cpu = utils_time_us();

  for (int nn = 0; nn < N; ++nn) {
    for (int mm = 0; mm < M; ++mm) {
      // mat_C_cpu[mm + M * nn] = 1.0 * (mm + 1) * (nn + 1) / M / N;
      int pos = mat_C_fpga_in_column_size * (nn / NUM_CH_C) + mm;
      // mat_C_fpga_in[nn % 8].resize(pos+1);
      mat_C_fpga_in[nn % NUM_CH_C][pos] = mat_C_cpu[mm + M * nn];
      // mat_C_fpga_in[0][pos] = mat_C_cpu[mm + M * nn];
      //  mat_C_cpu[mm ] = 0;
    }
  }

  preparing_C_time_cpu = (utils_time_us() - preparing_C_time_cpu) / (1e3);
  cout << "done\n";

  buffer_creation_time_cpu = utils_time_us();

  for (int i = 0; i < NUM_CH_SPARSE; i++) {
    OCL_CHECK(err,
              cl::Buffer currA(context,
                              CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sparse_A_fpga_column_size * sizeof(unsigned long),
                              sparse_A_fpga_vec[i].data(),
                              &err););
    buffer_A.push_back(std::move(currA));
  }

  for (int i = 0; i < NUM_CH_B; i++) {
    OCL_CHECK(err,
              cl::Buffer currA(context,
                              CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              mat_B_fpga_column_size * (N / NUM_CH_B) * sizeof(float),
                              mat_B_fpga_vec[i].data(),
                              &err););
    buffer_B.push_back(std::move(currA));
  }

  for (int i = 0; i < NUM_CH_C; i++) {
    OCL_CHECK(err,
              cl::Buffer currA(context,
                              CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              mat_C_fpga_in_chunk_size * sizeof(float),
                              mat_C_fpga_in[i].data(),
                              &err););
    buffer_C_in.push_back(std::move(currA));
  }

  if (enableP2P) {
    for (int i = 0; i < NUM_CH_C; i++) {
      cl_mem_ext_ptr_t buf_ext = {0};
      buf_ext.flags = XCL_MEM_EXT_P2P_BUFFER;
      OCL_CHECK(err,
                cl::Buffer currA(context,
                                CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
                                mat_C_fpga_chunk_size * sizeof(float),
                                &buf_ext,
                                &err););
      p2pbuffer_C.push_back(std::move(currA));
    }
  } else {
    for (int i = 0; i < NUM_CH_C; i++) {

      OCL_CHECK(err,
                cl::Buffer currA(context,
                                CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                mat_C_fpga_chunk_size * sizeof(float),
                                spmm_output_C_vec[i].data(),
                                &err););

      buffer_C.push_back(std::move(currA));
    }
  }

  OCL_CHECK(err,
            cl::Buffer buffer_edge_list_ptr(context,
                                           CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           edge_list_ptr_fpga_size * sizeof(unsigned int),
                                           edge_list_ptr_fpga.data(),
                                           &err););

  // set argument
  int parameter_pos = 0;

  OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, buffer_edge_list_ptr));

  for (int i = 0; i < NUM_CH_SPARSE; i++) {
    OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, buffer_A[i]));
  }

  for (int i = 0; i < NUM_CH_B; i++) {
    OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, buffer_B[i]));
  }

  for (int i = 0; i < NUM_CH_C; i++) {
    OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, buffer_C_in[i]));
  }

  if (enableP2P) {
    for (int i = 0; i < NUM_CH_C; i++) {
      OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, p2pbuffer_C[i]));
    }
  } else {
    for (int i = 0; i < NUM_CH_C; i++) {
      OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, buffer_C[i]));
    }
  }

  int MAX_SIZE_edge_LIST_PTR = edge_list_ptr.size() - 1;
  OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, MAX_SIZE_edge_LIST_PTR));

  int MAX_LEN_edge_PTR = edge_list_ptr[MAX_SIZE_edge_LIST_PTR];
  OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, MAX_LEN_edge_PTR));

  OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, M));
  OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, K));
  int para_N = (rp_time << 16) | N * 2;
  OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, para_N));

  unsigned int *tmpPointer_v;
  tmpPointer_v = (unsigned int *) &ALPHA;
  unsigned int alpha_int = *tmpPointer_v;
  tmpPointer_v = (unsigned int *) &BETA;
  unsigned int beta_int = *tmpPointer_v;
  OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, alpha_int));
  OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, beta_int));

  buffer_creation_time_cpu = (utils_time_us() - buffer_creation_time_cpu) / (1e3);

  for (int i = 0; i < NUM_CH_SPARSE; i++) {
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_A[i]}, 0 /* 0 means from host*/, NULL, &A_migration_events[i]));
  }

  OCL_CHECK(err,
            err = q.enqueueMigrateMemObjects({buffer_edge_list_ptr},
                                             0 /* 0 means from host*/,
                                             NULL,
                                             &Ptr_migration_event));

  for (int i = 0; i < NUM_CH_C; i++) {
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_C_in[i]},
                                               0 /* 0 means from host*/,
                                               NULL,
                                               &Cin_migration_events[i]));
  }
}

void FPGAHandler::setupP2PBuffers() {

  for (int i = 0; i < NUM_CH_C; i++) {
    cl_mem_ext_ptr_t buf_ext = {0};
    buf_ext.flags = XCL_MEM_EXT_P2P_BUFFER;
    OCL_CHECK(err,
              cl::Buffer currA(context,
                              CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
                              mat_C_fpga_chunk_size * sizeof(float),
                              &buf_ext,
                              &err););
    p2pbuffer_C.push_back(std::move(currA));
  }

  // set argument
  int parameter_pos = 0;

  // OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, buffer_edge_list_ptr));
  parameter_pos++;

  for (int i = 0; i < NUM_CH_SPARSE; i++) {
    // OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, buffer_A[i]));
    parameter_pos++;
  }

  for (int i = 0; i < NUM_CH_B; i++) {
    // OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, buffer_B[i]));
    parameter_pos++;
  }

  for (int i = 0; i < NUM_CH_C; i++) {
    // OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, buffer_C_in[i]));
    parameter_pos++;
  }

  for (int i = 0; i < NUM_CH_C; i++) {
    OCL_CHECK(err, err = krnl_sextans.setArg(parameter_pos++, p2pbuffer_C[i]));
  }
}

void FPGAHandler::executeSparseMultiplication(
    const SparseMatrix &sparse_matrix,
    const vector<float> &mat_B_cpu,
    const int &N,
    const float &ALPHA,
    const float &BETA) {

  int M = sparse_matrix.M;
  int K = sparse_matrix.K;
  int nnz = sparse_matrix.nnz;

  // cout << "Preparing dense B for FPGA ...";

  preparing_B_time_cpu = utils_time_us();

  if (NUM_CH_B == 4) {
#if 1
    for (int nn = 0; nn < N; ++nn) {
      for (int kk = 0; kk < K; ++kk) {
        int pos = (kk / 8) * 16 + (nn % 2) * 8 + kk % 8 + mat_B_fpga_column_size * (nn / 8);
        mat_B_fpga_vec[(nn / 2) % 4][pos] = mat_B_cpu[kk + K * nn];
      }
    }
#else
    for (int nn = 0; nn < N; ++nn)
        {
            for (int kk = 0; kk < K; ++kk)
            {
                int pos = (kk / 8) * 16 + (kk % 2) * 8 + nn % 8 + mat_B_fpga_column_size * (nn / 8);
                mat_B_fpga_vec[(kk / 2) % 4][pos] = mat_B_cpu[kk + K * nn];
            }
        }
#endif
  } else if (NUM_CH_B == 8) {
    for (int nn = 0; nn < N; ++nn) {
      for (int kk = 0; kk < K; ++kk) {
        int pos = kk + mat_B_fpga_column_size * (nn / 8);
        mat_B_fpga_vec[nn % 8][pos] = mat_B_cpu[kk + K * nn];
      }
    }
  }

  preparing_B_time_cpu = (utils_time_us() - preparing_B_time_cpu) / (1e3);

  // cout << "move data to device memory\n";

  B_moving_time_cpu = utils_time_us();

  for (int i = 0; i < NUM_CH_B; i++) {
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_B[i]}, 0 /* 0 means from host*/, NULL, &B_migration_events[i]));
  }

  q.finish();
  B_moving_time_cpu = (utils_time_us() - B_moving_time_cpu) / 1e3;

  int launch_num = 1;

  // printf("start kernel\n");
  auto start = std::chrono::steady_clock::now();
  OCL_CHECK(err, err = q.enqueueTask(krnl_sextans, NULL, &kernel_launch_event));
  q.finish();
  auto end = std::chrono::steady_clock::now();
  double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  time_taken *= 1e-9;

  // printf("Kernel time is %.7e ms\n", time_taken*1000/launch_num/rp_time);
  fpgaExecutionTime = time_taken * 1000 / launch_num / rp_time;

  float gflops =
      (2.0f * (nnz + M) * N) * launch_num // number of iterations of kernel launch
          * rp_time / 1e9                     // convert to GB
          / time_taken                        // total time in second
  ;
  // printf("GFLOPS:%f \n", gflops);
  fpgaGFLOPS = gflops;

  q.finish();
  // cout << "finish\n";
}

void FPGAHandler::transferOutputToCPU() {
  // cout << "move data to host\n";
  // Copy Result from Device Global Memory to Host Local Memory
  Cout_moving_time_cpu = utils_time_us();
  for (int i = 0; i < NUM_CH_C; i++) {
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_C[i]},
                                               CL_MIGRATE_MEM_OBJECT_HOST,
                                               NULL,
                                               &Cout_migration_events[i]));
  }
  q.finish();
  Cout_moving_time_cpu = (utils_time_us() - Cout_moving_time_cpu) / 1e3;
}

void FPGAHandler::mapFPGABufferToHost() {
  // Copy Result from FPGA Device Global Memory to GPU Device Global Memory

  for (int i = 0; i < NUM_CH_C; i++) {
    spmm_output_C_hostmap[i] = static_cast<float *>(q.enqueueMapBuffer(
        p2pbuffer_C[i],                        // buffer: The buffer object to be mapped
        CL_TRUE,                               // blocking_map: A blocking call (waits for the operation to complete)
        CL_MAP_READ | CL_MAP_WRITE,            // map_flags: Indicates we will be reading and writing
        0,                                     // offset: Buffer offset (start of the buffer)
        sizeof(float) * mat_C_fpga_chunk_size, // size: Size in bytes to be mapped
        nullptr,                               // event_wait_list: List of events to wait for (none in this case)
        &Cout_migration_events[i],             // event: Returns an event object that identifies this operation
        &err                                   // error code: Pointer to the error code variable
    ));
  }
}

void FPGAHandler::verifyFPGAResults(vector<float> &mat_C_cpu, int M, int N) {
  int mismatch_cnt = 0;

  for (int nn = 0; nn < N; ++nn) {
    for (int mm = 0; mm < M; ++mm) {
      float v_cpu = mat_C_cpu[mm + nn * M];
#if 1
      int pos = mat_C_fpga_column_size * (nn / NUM_CH_C) + mm;
      float v_fpga = spmm_output_C_vec[nn % NUM_CH_C][pos];
#else
      int pos = (mm / 8) * 16 + (mm % 2) * 8 + nn % 8 + mat_C_fpga_column_size * (nn / 8) + mat_C_fpga_chunk_size;
            float v_fpga = spmm_output_C_vec[(mm % 8) / 2][pos];
#endif

      // if(v_cpu!=v_fpga){
      // 	cout << "["<< nn <<"]["<< mm << "] v_cpu:" << v_cpu << " v_fpga:" <<v_fpga<< (v_cpu==v_fpga?"  incorrect!":"")<<endl;
      // }
      // else{
      // cout << "["<< nn <<"]["<< mm << "] v_cpu:" << v_cpu << " v_fpga:" <<v_fpga<< (v_cpu==v_fpga?"  correct!":"")<<endl;
      // }
      // printf("v_cpu: %f v_fpga: %f\n", v_cpu, v_fpga);

      float dff = fabs(v_cpu - v_fpga);
      float x = min(fabs(v_cpu), fabs(v_fpga)) + 1e-4;
      if (dff / x > 1e-3) { // 1e-4) {
        mismatch_cnt++;
      }
      // else{
      //     cout << "["<< nn <<"]["<< mm << "] v_cpu:" << v_cpu << " v_fpga:" <<v_fpga<< (v_cpu==v_fpga?"  exact match!":"")<<endl;
      // }
    }
  }

  float diffpercent = 100.0 * mismatch_cnt / M / N;
  bool pass = diffpercent < 2.0;

  if (pass) {
    cout << "Success!\n";
  } else {
    cout << "Failed.\n";
  }
  printf("num_mismatch = %d, percent = %.2f%%\n", mismatch_cnt, diffpercent);
}

void FPGAHandler::displayPerformanceMetrics() {

  // Start profiling timing

  computeEventsTime(B_partition_moving_time_ocl, B_moving_time_ocl, B_migration_events, "B (dense)", NUM_CH_B);
  computeEventsTime(A_partition_moving_time_ocl, A_moving_time_ocl, A_migration_events, "A (sparse)", NUM_CH_SPARSE);
  computeEventsTime(Cin_partition_moving_time_ocl, Cin_moving_time_ocl, Cin_migration_events, "Cin", NUM_CH_C);
  computeEventsTime(Cout_partition_moving_time_ocl, Cout_moving_time_ocl, Cout_migration_events, "Cout", NUM_CH_C);

  // For kernel events
  uint64_t nstimestart, nstimeend;

  OCL_CHECK(err,
            err = kernel_launch_event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
  OCL_CHECK(err,
            err = kernel_launch_event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
  kernel_time_ocl = (nstimeend - nstimestart) / 1e6;

  cinThroughput = (sizeof(float) * mat_C_fpga_chunk_size * NUM_CH_C / 1e9) / (Cin_moving_time_ocl / 1000);
  double cinSize = sizeof(float) * mat_C_fpga_chunk_size * NUM_CH_C / 1e6;
  coutThroughput = (sizeof(float) * mat_C_fpga_chunk_size * NUM_CH_C / 1e9) / (f2g_p2p_datamovement_time / 1000);

  std::cout << std::fixed << std::setprecision(3);

  std::cout << "==============================================" << std::endl;
  std::cout << "      FPGA PERFORMANCE METRICS" << std::endl;
  std::cout << "==============================================" << std::endl
            << std::endl;

  std::cout << "------ SETUP TIMINGS ------" << std::endl;
  std::cout << "Preparation - SparseA:          " << preparing_A_time_cpu << " msec" << std::endl;
  std::cout << "Preparation - C:                " << preparing_C_time_cpu << " msec" << std::endl;
  std::cout << "A - Avg Move Time/Buffer:       " << A_partition_moving_time_ocl << " msec     | Elapsed: "
            << A_moving_time_ocl << " msec" << std::endl;
  std::cout << "Cin - Avg Move Time/Buffer:     " << Cin_partition_moving_time_ocl << " msec    | Elapsed: "
            << Cin_moving_time_ocl << " msec" << std::endl
            << std::endl;

  std::cout << "------ FPGA SPDMM TIMINGS (CPU PERSPECTIVE) ------" << std::endl;
  std::cout << "FPGA SPDMM:                     " << fpga_spdmm_sextans_time << " msec" << std::endl
            << std::endl;

  std::cout << "Preparation - Dense B:          " << preparing_B_time_cpu << " msec" << std::endl;
  std::cout << "Dense B Move Time:              " << B_moving_time_cpu << " msec" << std::endl;
  std::cout << "Kernel Execution Time:          " << fpgaExecutionTime << " msec" << std::endl;
  std::cout << "Output Buffer Move:             " << Cout_moving_time_cpu << " msec" << std::endl
            << std::endl;

  std::cout << "------ FPGA SPDMM TIMINGS (OPENCL EVENTS) ------" << std::endl;
  std::cout << "Dense B - Avg Move Time/Buffer: " << B_partition_moving_time_ocl << " msec     | Elapsed: "
            << B_moving_time_ocl << " msec" << std::endl;
  std::cout << "Cout - f2c Avg Move Time/Buffer:" << Cout_partition_moving_time_ocl << " msec  | Elapsed: "
            << Cout_moving_time_ocl << std::endl
            << std::endl;
  std::cout << "Cout - f2g (p2p) elapsed Time:  " << f2g_p2p_datamovement_time << std::endl;
  std::cout << "Kernel Execution Time:          " << kernel_time_ocl << " msec" << std::endl
            << std::endl;

  std::cout << "------ DATA THROUGHPUT & SIZE ------" << std::endl;
  std::cout << "Cin Throughput:                 " << cinThroughput << " GB/s      | Size: " << cinSize << " MB"
            << std::endl;
  std::cout << "Cout Throughput (f2g, p2p):     " << coutThroughput << " GB/s" << std::endl
            << std::endl;

  std::cout << "------ CPU METRICS ------" << std::endl;
  std::cout << "CPU Execution Time:             " << cpuExecutionTime << " msec" << std::endl;
  std::cout << "CPU GFLOPS:                     " << cpuGFLOPS << std::endl
            << std::endl;

  std::cout << "------ FPGA METRICS ------" << std::endl;
  std::cout << "FPGA Kernel Execution Time:     " << fpgaExecutionTime << " msec" << std::endl;
  std::cout << "FPGA GFLOPS:                    " << fpgaGFLOPS << std::endl;
}

void FPGAHandler::initializePerformanceMetrics() {
  preparing_A_time_cpu = 0;
  preparing_B_time_cpu = 0;
  preparing_C_time_cpu = 0;
  buffer_creation_time_cpu = 0;
  kernel_creation_time = 0;
  B_moving_time_cpu = 0;
  kernel_launch_time = 0;
  Cout_moving_time_cpu = 0;
  f2g_p2p_datamovement_time = 0;

  B_partition_moving_time_ocl = 0;
  A_partition_moving_time_ocl = 0;
  Cin_partition_moving_time_ocl = 0;
  Cout_partition_moving_time_ocl = 0;
  kernel_time_ocl = 0;

  B_moving_time_ocl = 0;
  A_moving_time_ocl = 0;
  Cin_moving_time_ocl = 0;
  Cout_moving_time_ocl = 0;

  cinThroughput = 0;
  coutThroughput = 0;
}

// Return spdmm total time, B prepare time, B move time, Execution time, Output move time
std::string FPGAHandler::csvUpdateSPMMTime() {
  std::ostringstream ss;
  ss << fpga_spdmm_sextans_time << ","
     << preparing_B_time_cpu << ","
     << B_moving_time_cpu << ","
     << fpgaExecutionTime << ","
     << Cout_moving_time_cpu;
  return ss.str();
}

// Return
std::string FPGAHandler::csvUpdateSetupTime() {
  std::ostringstream ss;
  ss << preparing_A_time_cpu << ","
     << preparing_C_time_cpu << ","
     << A_moving_time_ocl << ","
     << Cin_moving_time_ocl;
  return ss.str();
}

std::string FPGAHandler::csvUpdateAllTime() {
  std::ostringstream ss;
  ss << fpga_spdmm_sextans_time << ","
     << preparing_B_time_cpu << ","
     << B_moving_time_cpu << ","
     << fpgaExecutionTime << ","
     << Cout_moving_time_cpu << ","
     << B_moving_time_ocl << ","
     << Cout_moving_time_ocl << ","
     << f2g_p2p_datamovement_time << ","
     << kernel_time_ocl << ","
     << cinThroughput << ","
     << coutThroughput << ","
     << cpuExecutionTime << ","
     << cpuGFLOPS << ","
     << fpgaExecutionTime << ","
     << fpgaGFLOPS;
  return ss.str();
}

double FPGAHandler::calculateAveragePower(const std::string &filename) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return -1; // Return -1 as an indication of error
  }

  // Skip the first line (header)
  std::string line;
  std::getline(file, line);

  // Skip the second line (column names)
  std::getline(file, line);

  double totalPower = 0;
  int rowCount = 0;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string token;

    // Read and ignore timestamp
    std::getline(ss, token, ',');

    // Read 12v_aux_curr
    std::getline(ss, token, ',');
    double aux_curr = std::stod(token);

    // Read 12v_aux_vol
    std::getline(ss, token, ',');
    double aux_vol = std::stod(token);

    // Read 12v_pex_curr
    std::getline(ss, token, ',');
    double pex_curr = std::stod(token);

    // Read 12v_pex_vol
    std::getline(ss, token, ',');
    double pex_vol = std::stod(token);
    // cout << aux_curr << "," <<aux_vol <<","<< pex_curr << ","<< pex_vol << endl;

    totalPower += (aux_curr * aux_vol + pex_curr * pex_vol) / 1000000;
    rowCount++;
  }

  if (rowCount == 0) {
    return 0; // Avoid division by zero
  }

  return totalPower / rowCount;
}
