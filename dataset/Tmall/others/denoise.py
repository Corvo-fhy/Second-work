import json
from pathlib import Path

# 设置路径
data_folder = Path('./')  # 当前目录
jsonl_path = data_folder / 'noise-items.jsonl'

# 要处理的文件
target_files = ['click.txt', 'cart.txt']

# Step 1: 加载硬噪声信息：custom_id -> set(item_id)
hard_noise_map = {}
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        cid = data.get('custom_id')
        hard_ids = set(data.get('hard_noise_ids', []))
        if hard_ids:
            hard_noise_map[cid] = hard_ids

# Step 2: 清洗数据 + 统计信息
for fname in target_files:
    file_path = data_folder / fname
    if not file_path.exists():
        print(f"[跳过] 文件不存在：{fname}")
        continue

    cleaned_path = data_folder / fname.replace('.txt', '_cleaned.txt')
    total_removed = 0
    affected_users = 0

    with open(file_path, 'r', encoding='utf-8') as fin, open(cleaned_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            cid, item_ids = parts[0], parts[1:]
            removed_count = 0

            if cid in hard_noise_map:
                original_len = len(item_ids)
                item_ids = [item for item in item_ids if item not in hard_noise_map[cid]]
                removed_count = original_len - len(item_ids)
                if removed_count > 0:
                    affected_users += 1
                    total_removed += removed_count

            if item_ids:
                fout.write(cid + ' ' + ' '.join(item_ids) + '\n')
            else:
                fout.write(cid + '\n')

    # 输出统计信息
    print(f"[处理完成] {fname}")
    print(f" - 删除硬噪声 item 数量: {total_removed}")
    print(f" - 影响的 custom_id 数量: {affected_users}")
    print(f" - 清洗后保存为: {cleaned_path.name}\n")
