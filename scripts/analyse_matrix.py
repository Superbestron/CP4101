import sys
import heapq

# This script analyses the distribution of non-zeros of a matrix across R % P.
# Matrix must be in CSR format.
# Usage: python3 analyse_matrix.py <matrix file name> <optionally: P (default is R)>

P = -1
mtx_file_path = sys.argv[1]
TOP_NUM = 10


def read_mtx(file_path):
    global P
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # detect if pattern or coordinate
    header = lines[0]
    isPattern = "pattern" in header
    isSymmetric = "symmetric" in header

    # Skip comments
    lines = [line for line in lines if not line.startswith('%')]

    # Extract matrix properties
    rows, cols, _ = map(int, lines[0].split())

    # Extract non-zero entries
    if isPattern:
        entries = [(int(line.split()[0]), int(line.split()[1])) for line in lines[1:]]
    else:
        entries = [(int(line.split()[0]), int(line.split()[1]), float(line.split()[2])) for line in lines[1:]]

    P = rows
    if len(sys.argv) == 3:
        P = int(sys.argv[2])
    return rows, cols, entries, isPattern, isSymmetric


def analyze_distribution(entries, isPattern, isSymmetric):
    row_modulo_counts = [0] * P
    dict = set()
    non_empty_rows = set()

    for entry in entries:
        if isPattern:
            row, col = entry
        else:
            row, col, value = entry
            if value == 0:
                continue
        if (row, col) not in dict:
            row_modulo_counts[row % P] += 1
            dict.add((row, col))
            non_empty_rows.add(row)
            if isSymmetric and (col, row) not in dict:
                row_modulo_counts[col % P] += 1
                dict.add((col, row))
                non_empty_rows.add(col)

    heap = []
    max_row_nnz = 0

    for i, count in enumerate(row_modulo_counts):
        max_row_nnz = max(max_row_nnz, count)
        heapq.heappush(heap, (count, -i))
        if len(heap) > TOP_NUM:
            heapq.heappop(heap)

    ls = [heapq.heappop(heap) for _ in range(len(heap))]
    ls.reverse()

    for cnt, idx in ls:
        if cnt > 0:
            print(f"Index {P} % {-idx}: {cnt} NNZs, {100 * cnt / len(dict):.2f}%")

    print(f"{len(non_empty_rows)} Non empty rows")

    return max_row_nnz, len(dict)


if __name__ == "__main__":
    rows, cols, entries, isPattern, isSymmetric = read_mtx(mtx_file_path)

    print(f"Analysis of Top {TOP_NUM} Row Non-Zero Modulo {P} Distribution:")
    max_row_nnz, nnz = analyze_distribution(entries, isPattern, isSymmetric)
    print(f"\nMatrix properties: Rows={rows}, Columns={cols}, Non-zero entries={nnz}")
    print(f"Maximum NNZs per PU/Wavefront: {max_row_nnz}")
