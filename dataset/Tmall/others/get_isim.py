import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix, save_npz
from pathlib import Path
import random
from tqdm import tqdm

data_path = Path('./train.txt')

# Step 1: 读取 train 数据
user_items = []
item_to_users = defaultdict(set)

# 用于映射物品ID到矩阵的行列索引
item_id_to_index = {}

with open(data_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        parts = line.strip().split()
        if not parts:
            continue
        cid, items = parts[0], parts[1:]
        user_items.append(set(items))
        for item in items:
            item_to_users[item].add(idx)

# 为每个物品分配一个唯一的行列索引
all_items = sorted(item_to_users.keys())  # 物品按照ID排序
item_id_to_index = {item: i for i, item in enumerate(all_items)}  # 映射物品ID到行列索引

# 自定义矩阵的大小（30316个物品）
num_items = 30316
# Step 2: 创建一个固定维度的稀疏矩阵，初始化为0
co_matrix = lil_matrix((num_items, num_items), dtype=np.int32)

# Step 3: 构建二进制共现矩阵（共现为1，不共现为0）
print("计算物品共现矩阵...")
for item, users in tqdm(item_to_users.items(), desc="物品共现统计", unit="物品"):
    item_index = item_id_to_index[item]  # 获取物品对应的矩阵索引
    users = list(users)
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1, u2 = users[i], users[j]
            co_matrix[item_index, item_index] = 1  # 物品与自己共现
            # 更新共现矩阵
            co_matrix[item_index, item_index] = 1
            co_matrix[item_index, item_index] = 1

# Step 4: 处理没有共现的物品，如果没有与其他物品共现，随机选择一个物品作为共现
print("处理没有共现的物品...")
for i in tqdm(range(num_items), desc="检查未共现物品", unit="物品"):
    # 检查当前物品是否有与其他物品共现
    if co_matrix[i].sum() == 0:
        # 如果没有共现，则随机选择一个物品进行共现
        random_item = random.choice(all_items)
        co_matrix[i, item_id_to_index[random_item]] = 1
        co_matrix[item_id_to_index[random_item], i] = 1  # 保证对称

# Step 5: 保存物品共现矩阵
output_path = Path('./item_cooccurrence_matrix_with_random.npz')
save_npz(output_path, co_matrix.tocsr())

# 输出共现矩阵的维度
print(f"[完成] 物品共现矩阵已计算，维度：{co_matrix.shape}，保存路径：{output_path}")
