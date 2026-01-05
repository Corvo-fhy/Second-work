import numpy as np


data = np.load('user_item_similarity_matrix.npz')
similarity_matrix = data['similarity_matrix']
print(similarity_matrix.shape)  # 应该是 (8302, 30316)

# 查看某一行
i = 0
user_sim = similarity_matrix[i]
valid_sim = user_sim[user_sim >= 0]
print(f"User {i} max similarity: {np.max(valid_sim)}, min similarity: {np.min(valid_sim)}")
