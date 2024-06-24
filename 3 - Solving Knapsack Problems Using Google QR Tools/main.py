from ortools.algorithms.python import knapsack_solver
import os
import csv
import time
from itertools import islice
import matplotlib.pyplot as plt

# File ghi kết quả
csv_file_path = 'result.csv'

# Các đường dẫn để đọc dữ liệu
testGroups = ['n00050', 'n00100', 'n00200', 'n00500', 'n01000']
RTestGroups = ['R01000', 'R10000']
groupNames = ['00Uncorrelated', '01WeaklyCorrelated', '02StronglyCorrelated', '03InverseStronglyCorrelated',
             '04AlmostStronglyCorrelated', '05SubsetSum', '06UncorrelatedWithSimilarWeights', '07SpannerUncorrelated',
              '08SpannerWeaklyCorrelated', '09SpannerStronglyCorrelated', '10MultipleStronglyCorrelated',
              '11ProfitCeiling', '12Circle']
filenames = ['s019.kp'] # Chọn file 019 để chạy

with open(csv_file_path, mode='w', newline='') as csv_file:
    # Chuẩn bị để ghi kết quả vào file
    writer = csv.writer(csv_file)
    writer.writerow(['Group Name', 'Size', 'Range', 'Filename', 'Total Value', 'Total Weight', 'Time', 'isOptimal'])

    # Lặp qua file 019 của tất cả các nhóm và trường hợp
    for groupName in groupNames:
        for testGroup in testGroups:
            for RTestGroup in RTestGroups:
                for filename in filenames:
                    filepath = f"kplib/{groupName}/{testGroup}/{RTestGroup}/{filename}"
                    if not os.path.exists(filepath): # nếu không tồn tại file thì tiếp tục vòng lặp tiếp theo
                        print("File does not exist")
                        continue

                    # Khởi tạo mảng capacities (sức chứa), values (giá trị), và weights (khối lượng)
                    capacities = []
                    values = []
                    weights = []

                    # Đọc dữ liệu từ file input, output là mảng với mỗi phần tử là một dòng trong file
                    with open(filepath, 'r') as f:
                        lines = f.read().splitlines()

                    # Ghi lại sức chứa từ dữ liệu đã cho
                    capacities.append(int(lines[2]))

                    # Lặp qua từng dòng từ dòng thứ 4 trở đi, giá trị là phần tử đàu tiên, khối lượng là phần tử thứ 2
                    for line in islice(lines, 4, None):
                        data = line.split()
                        values.append(int(data[0]))
                        weights.append(int(data[1]))

                    # Khởi tạo solver knapsack từ or tool
                    solver = knapsack_solver.KnapsackSolver(
                        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
                        "KnapsackExample",
                    )

                    # Đặt giá trị time limit là 200s
                    time_limit = 200

                    # Đưa các giá trị cần thiết vào solver và bắt đầu giải bài toán
                    solver.init(values, weights, capacities)
                    solver.set_time_limit(time_limit)
                    start = time.perf_counter()
                    computed_value = solver.solve()
                    end = time.perf_counter()

                    # Thời gian giải bài toán bằng thời gian kết thúc trừ đi thời gian bắt đầu
                    elapsed_time = end - start

                    # Ghi lại các món đồ được chọn và tổng khối lượng của chúng
                    packed_weights = [weights[0][i] for i in range(len(values)) if solver.best_solution_contains(i)]
                    total_weight = sum(packed_weights)

                    # Kiểm tra tối ưu bằng cách gọi hàm is_solution_optimal có sẵn
                    isOptimal = "Yes" if solver.is_solution_optimal() else "No"

                    # Ghi lại kết quả và in ra console
                    writer.writerow([groupName, testGroup, RTestGroup, filename, computed_value, total_weight, elapsed_time, isOptimal])

                    print(f'Group: {groupName} | Size: {testGroup} | Range: {RTestGroup} | File: {filename} completed. Time: {elapsed_time:.8f} seconds, Is Optimal: {isOptimal}')

                    # # Visualize các dataset
                    # plt.title(f"{groupName}")
                    # plt.plot(weights, values, "o")
                    # plt.savefig(f"{groupName}{testGroup}{RTestGroup}.png")
