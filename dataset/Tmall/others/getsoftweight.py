import json
import os
import numpy as np
from tqdm import tqdm

# 配置路径
user_embedding_dir = '/home/fanhongyu/fhy_data/LLM4Rec/embeddings/User_emb'  # 修改成你的用户embedding文件夹
item_embedding_dir = '/home/fanhongyu/fhy_data/LLM4Rec/embeddings/Item_emb'  # 修改成你的物品embedding文件夹
noise_jsonl_path = '/home/fanhongyu/fhy_data/LLM4Rec/result/noise-items.jsonl'            # 你的噪声jsonl文件
output_path = 'user_item_similarity_matrix.npz'  # 输出保存路径

user_num = 8302
item_num = 30316

# 1. 读取 jsonl，提取 soft_noise
user_soft_noise = {}  # user_id -> list of soft_noise_ids

with open(noise_jsonl_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        user_id = int(data['custom_id'])  # int型
        soft_noise_ids = data.get('soft_noise_ids', [])
        if soft_noise_ids:
            user_soft_noise[user_id] = list(map(int, soft_noise_ids))

# 2. 初始化全尺寸 similarity 矩阵
similarity_matrix = np.ones((user_num, item_num), dtype=np.float32)  # 初始化为1

# 3. 计算余弦相似度
def load_embedding(embedding_dir, id_):
    path = os.path.join(embedding_dir, f'{id_}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Embedding file not found: {path}')
    return np.load(path)

print("Calculating similarities...")
for user_id in tqdm(range(user_num), desc="Processing users"):
    if user_id not in user_soft_noise:
        continue  # 这个用户没有 soft noise，不处理
    
    try:
        user_emb = load_embedding(user_embedding_dir, user_id)
    except FileNotFoundError:
        print(f"Warning: user {user_id} embedding not found, skipping.")
        continue

    for item_id in user_soft_noise[user_id]:
        if item_id < 0 or item_id >= item_num:
            print(f"Warning: item id {item_id} out of range, skipping.")
            continue
        try:
            item_emb = load_embedding(item_embedding_dir, item_id)
        except FileNotFoundError:
            print(f"Warning: item {item_id} embedding not found, skipping.")
            continue

        # 计算余弦相似度
        num = np.dot(user_emb, item_emb)
        denom = np.linalg.norm(user_emb) * np.linalg.norm(item_emb)
        if denom == 0:
            sim = 0.0
        else:
            sim = num / denom
        
        # 归一化到 [0,1]
        sim = (sim + 1.0) / 2.0

        # 保存
        similarity_matrix[user_id, item_id] = sim

# 4. 保存结果
np.savez(output_path, similarity_matrix=similarity_matrix)

print(f"\n✅ Saved full similarity matrix to {output_path}, shape: {similarity_matrix.shape}")