import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix, save_npz
from pathlib import Path

data_path = Path('./train.txt')

# Step 1: 读取 train 数据
user_items = []
user_id_map = {}  # 自定义 id 映射为连续数字索引
item_to_users = defaultdict(set)

with open(data_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        parts = line.strip().split()
        if not parts:
            continue
        cid, items = parts[0], parts[1:]
        user_id_map[cid] = idx  # 映射 custom_id 到矩阵行索引
        user_items.append(set(items))
        for item in items:
            item_to_users[item].add(idx)

num_users = len(user_items)
co_matrix = lil_matrix((num_users, num_users), dtype=np.int32)

# Step 2: 用户共现统计
for item, users in item_to_users.items():
    users = list(users)
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1, u2 = users[i], users[j]
            co_matrix[u1, u2] = 1  # 只要共现就设置为 1
            co_matrix[u2, u1] = 1  # 保证对称

# Step 3: 保存稀疏矩阵
output_path = Path('./user_cooccurrence_matrix.npz')
save_npz(output_path, co_matrix.tocsr())

print(f"[完成] 用户共现矩阵 shape={co_matrix.shape}，非零元素={co_matrix.nnz}")
print(f"[保存] 路径：{output_path}")
