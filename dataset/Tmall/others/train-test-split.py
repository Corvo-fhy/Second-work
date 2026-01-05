import random

# 输入和输出文件路径
input_file = "alipay.txt"
train_file = "train.txt"
test_file = "test.txt"

# 读取原始数据并按用户分组
user_interactions = {}

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        parts = line.strip().split()
        user_id = parts[0]
        items = parts[1:]
        
        user_interactions[user_id] = items

# 分割数据集
with open(train_file, "w", encoding="utf-8") as train_outfile, open(test_file, "w", encoding="utf-8") as test_outfile:
    for user_id, items in user_interactions.items():
        if len(items) > 1:
            # 随机选择一个物品作为测试集
            test_item = random.choice(items)
            items.remove(test_item)
            
            # 训练集包括用户的其他物品
            train_outfile.write(f"{user_id} " + " ".join(items) + "\n")
            
            # 测试集包括用户选择的物品
            test_outfile.write(f"{user_id} {test_item}\n")
        elif len(items) == 1:
            # 只有一个交互物品时，将其作为测试集，并不产生训练集数据
            test_outfile.write(f"{user_id} {items[0]}\n")

print("数据集已分割，训练集和测试集已保存。")
