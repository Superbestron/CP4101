import sys
import heapq

# This script reorders the rows of the matrix to minimise the maximum NNZ in each R % P.
# Matrix must be in CSR format and non-symmetric.
# Usage: python3 reorder_matrix.py <matrix file name>
# The default P for GPU and FPGA respectively is 64 and 80 respectively, you may change
# it in the arguments to reorder_rows().

mtx_file_path = sys.argv[1]
mtx_gpu_new_file_path = f"{mtx_file_path[:-4]}_re_gpu.mtx"
mtx_fpga_new_file_path = f"{mtx_file_path[:-4]}_re_fpga.mtx"


def read_mtx(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # detect if pattern or coordinate
    header = lines[0]
    isPattern = "pattern" in header
    if "symmetric" in header:
        print("Convert matrix to non-symmetric format first!")
        assert False

    # Skip comments
    arr = []
    with open(mtx_gpu_new_file_path, 'w') as file:
        for line in lines:
            if line.startswith('%'):
                file.write(line)
            else:
                arr.append(line)
        file.write("%Reordered Matrix\n")

    with open(mtx_fpga_new_file_path, 'w') as file:
        for line in lines:
            if line.startswith('%'):
                file.write(line)
            else:
                break
        file.write("%Reordered Matrix\n")

    lines = arr

    # Extract matrix properties
    rows, cols, _ = map(int, lines[0].split())

    # Extract non-zero entries
    if isPattern:
        entries = [(int(line.split()[0]), int(line.split()[1])) for line in lines[1:]]
    else:
        entries = [(int(line.split()[0]), int(line.split()[1]), float(line.split()[2])) for line in lines[1:]]

    return rows, cols, entries, isPattern


def reorder_rows(rows, cols, entries, isPattern, P, mtx_new_file_path):
    row_entries = [[] for i in range(rows)]

    nnz = 0
    # preprocess into rows
    for entry in entries:
        if isPattern:
            row, col = entry
            row_entries[row - 1].append(col)
        else:
            row, col, value = entry
            if value == 0:
                continue
            row_entries[row - 1].append((col, value))
        nnz += 1

    print(f"{nnz} NNZs")

    with open(mtx_new_file_path, 'a') as file:
        file.write(f"{rows} {cols} {nnz}\n")

    row_entries.sort(key=lambda x: len(x), reverse=True)

    class Processor:
        def __init__(self, size, rows_taken, idx):
            self.size = size
            self.rows_taken = rows_taken
            self.idx = idx

        def __lt__(self, other):
            ratio_self = self.size
            ratio_other = other.size
            return ratio_self < ratio_other

    lim = (rows + (P - 1)) // P

    P_arr = [[] for _ in range(P)]
    heap = []
    heapq.heapify(heap)
    remainder = rows % P
    if remainder:
        # remainder P can hold extra
        for i in range(remainder):
            heapq.heappush(heap, Processor(0, 0, i))  # (size, rows_taken, idx)
        for i in range(remainder, P):
            heapq.heappush(heap, Processor(0, 1, i))  # (size, rows_taken, idx)
    else:
        for i in range(P):
            heapq.heappush(heap, Processor(0, 0, i))  # (size, rows_taken, idx)

    for row_no, row_entry in enumerate(row_entries):
        length = len(row_entry)
        wavefront = heapq.heappop(heap)
        if wavefront.rows_taken + 1 < lim:  # if wavefront is full, remove from heap
            heapq.heappush(heap, Processor(wavefront.size + length, wavefront.rows_taken + 1, wavefront.idx))
        P_arr[wavefront.idx].append(row_no)

    load_factors = []

    # retrieve rows from wavefronts and write to file
    with open(mtx_new_file_path, 'a') as file:
        for wave_idx, wavefront in enumerate(P_arr):
            cnt = 0
            for actual_row_no, row_no in enumerate(wavefront):
                for entry in row_entries[row_no]:
                    if isPattern:
                        col = entry
                        file.write(f"{wave_idx + actual_row_no * P + 1} {col}\n")
                    else:
                        col, value = entry
                        file.write(f"{wave_idx + actual_row_no * P + 1} {col} {value}\n")
                    cnt += 1
            load_factors.append(cnt)

    load_factors.sort(reverse=True)
    print("Top 10 PE Loads:")
    for i in range(10):
        print(f"  {i + 1}. {load_factors[i]}")


if __name__ == "__main__":
    rows, cols, entries, isPattern = read_mtx(mtx_file_path)

    reorder_rows(rows, cols, entries, isPattern, 64, mtx_gpu_new_file_path)
    print(f"File {mtx_gpu_new_file_path} written!")

    reorder_rows(rows, cols, entries, isPattern, 64, mtx_fpga_new_file_path)
    print(f"File {mtx_fpga_new_file_path} written!")
