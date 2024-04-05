import sys

# This script converts a CSR symmetric matrix to a CSR non-symmetric matrix.
# Matrix must be in CSR format.
# Usage: python3 convert_to_non_symmetric_py <matrix file name>

mtx_file_path = sys.argv[1]
mtx_new_file_path = f"{mtx_file_path[:-4]}_nonsy.mtx"


def read_mtx(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # detect if pattern or coordinate
    header = lines[0][:-10]  # remove symmetric
    isPattern = "pattern" in header

    with open(mtx_new_file_path, 'w') as file:
        file.write(f"{header}general\n")

    lines = lines[1:]

    # Skip comments
    arr = []
    with open(mtx_new_file_path, 'a') as file:
        for line in lines:
            if line.startswith('%'):
                file.write(line)
            else:
                arr.append(line)

    lines = arr

    # Extract matrix properties
    rows, cols, _ = map(int, lines[0].split())

    # Extract non-zero entries
    if isPattern:
        entries = [(int(line.split()[0]), int(line.split()[1])) for line in lines[1:]]
    else:
        entries = [(int(line.split()[0]), int(line.split()[1]), float(line.split()[2])) for line in lines[1:]]

    dict = {}
    for entry in entries:
        value = 0
        if isPattern:
            row, col = entry
        else:
            row, col, value = entry
            if value == 0:
                continue
        if (row, col) not in dict:
            dict[(row, col)] = value
            if (col, row) not in dict:
                dict[(col, row)] = value

    print(f"{len(dict)} NNZs")

    with open(mtx_new_file_path, 'a') as file:
        file.write(f"{rows} {cols} {len(dict)}\n")
        for (row, col), value in dict.items():
            if isPattern:
                file.write(f"{row} {col}\n")
            else:
                file.write(f"{row} {col} {value}\n")


if __name__ == "__main__":
    read_mtx(mtx_file_path)
    print(f"File {mtx_new_file_path} written!")
