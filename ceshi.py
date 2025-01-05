import random
import numpy as np
import time
from scipy.stats import qmc

# 定义子函数
def sphere_function(x):
    return sum([xi**2 for xi in x])

def rastrigin_function(x):
    return 10 * len(x) + sum([xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x])

def griewank_function(x):
    sum_term = sum([xi**2 for xi in x]) / 4000
    prod_term = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    return sum_term - prod_term + 1

# 定义复合函数
def composite_function(x, o1, o2, o3, T1, T2, T3, B=0):
    epsilon = 1e-6
    w1 = 1 / np.sqrt(np.sum((x - o1)**2) + epsilon)
    w2 = 1 / np.sqrt(np.sum((x - o2)**2) + epsilon)
    w3 = 1 / np.sqrt(np.sum((x - o3)**2) + epsilon)
    
    term1 = w1 * sphere_function(T1 @ (x - o1))
    term2 = w2 * rastrigin_function(T2 @ (x - o2))
    term3 = w3 * griewank_function(T3 @ (x - o3))
    
    return term1 + term2 + term3 + B

# FDPS优化算法
def fdps(objective_function, bounds, num_partitions=5000, num_samples=10, tolerance=1e-6, max_time=300):
    dim = len(bounds)
    current_bounds = np.array(bounds)
    best_solution = None
    best_value = float('inf')

    start_time = time.time()
    previous_best_value = float('inf')
    iteration = 0

    while True:
        iteration += 1

        # 检查时间限制
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            print(f"Terminated after reaching max time: {max_time} seconds.")
            break

        # 划分域
        partitions = [np.linspace(current_bounds[i, 0], current_bounds[i, 1], num=num_partitions) for i in range(dim)]
        partitions = np.array(partitions)

        subdomains = []
        for i in range(partitions.shape[1] - 1):  # 迭代区间
            subdomain_lower = partitions[:, i]
            subdomain_upper = partitions[:, i + 1]
            subdomains.append((subdomain_lower, subdomain_upper))

        # 在每个子域中评估样本并跟踪其性能
        subdomain_performance = []
        for lower, upper in subdomains:
            sampler = qmc.LatinHypercube(d=dim)
            samples = qmc.scale(sampler.random(num_samples), lower, upper)
            subdomain_best_value = float('inf')
            for sample in samples:
                value = objective_function(sample)
                if value < best_value:
                    best_solution, best_value = sample, value
                if value < subdomain_best_value:
                    subdomain_best_value = value
            subdomain_performance.append((subdomain_best_value, lower, upper))

        # 按最佳性能排序子域
        subdomain_performance.sort(key=lambda x: x[0])

        # 为表现较好的子域分配更多样本
        num_samples_per_subdomain = [num_samples] * len(subdomains)
        for i in range(len(subdomains) // 2):
            num_samples_per_subdomain[i] *= 2

        # 检查收敛性
        if abs(previous_best_value - best_value) < tolerance:
            print(f"Converged after {iteration} iterations in {elapsed_time:.2f} seconds.")
            break
        previous_best_value = best_value

        # 缩小搜索空间
        new_bounds = []
        for i in range(dim):
            center = (current_bounds[i, 0] + current_bounds[i, 1]) / 2
            if best_solution[i] <= center:
                new_bounds.append((current_bounds[i, 0], center))
            else:
                new_bounds.append((center, current_bounds[i, 1]))
        current_bounds = np.array(new_bounds)

    return best_solution, best_value, elapsed_time

# 参数
bounds = [(-10, 10)] * 5  # 5维问题
o1 = np.random.uniform(-10, 10, 5)
o2 = np.random.uniform(-10, 10, 5)
o3 = np.random.uniform(-10, 10, 5)
T1 = np.eye(5)
T2 = np.eye(5)
T3 = np.eye(5)

# 运行FDPS优化算法30次并收集结果
results = []
for _ in range(30):
    best_solution, best_value, elapsed_time = fdps(lambda x: composite_function(x, o1, o2, o3, T1, T2, T3), bounds)
    results.append(best_value)

average_value = np.mean(results)
standard_deviation_value = np.std(results)

print(f"Average Fitness Value: {average_value}")
print(f"Variance: {standard_deviation_value}")