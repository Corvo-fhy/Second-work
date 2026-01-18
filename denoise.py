
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-12)

def load_embedding(embed_dir, idx):
    path = os.path.join(embed_dir, f"{idx}.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    else:
        raise FileNotFoundError(f"Embedding not found: {path}")

def denoise_behavior_file(
    filepath,
    user_emb_dir,
    item_emb_dir,
    remove_ratio=0.2,
    temperature=5.0,
    similarity_threshold=0.2,  
    min_keep=1
):
    denoised_lines = []
    total_removed = 0

    with open(filepath, 'r') as f:
        lines = f.readlines()

    print(f"Processing {os.path.basename(filepath)} ({len(lines)} lines)...")

    for line in tqdm(lines, desc=f"Denoising {os.path.basename(filepath)}", unit="line"):
        stripped = line.strip()
        if not stripped:
            denoised_lines.append("")
            continue

        parts = list(map(int, stripped.split()))
        uid = parts[0]
        items = parts[1:]
        if len(items) <= min_keep:
            denoised_lines.append(stripped)
            continue

        try:
            u_emb = load_embedding(user_emb_dir, uid).reshape(1, -1)
        except Exception:
            denoised_lines.append(stripped)
            continue

        sims = []
        valid_items = []
        for item in items:
            try:
                v_emb = load_embedding(item_emb_dir, item).reshape(1, -1)
                sim = cosine_similarity(u_emb, v_emb)[0, 0]
                sims.append(sim)
                valid_items.append(item)
            except Exception:
                sims.append(-1e5)
                valid_items.append(item)

        M = len(sims)
        sims_arr = np.array(sims)

        # === 关键逻辑：只删低相似度项，且不超过 remove_ratio 上限 ===
        low_sim_indices = np.where(sims_arr < similarity_threshold)[0]

        # 可选：在低相似度项中，优先删最不相关的（按 softmax 概率）
        if len(low_sim_indices) > 0:
            # 对低相似度项计算删除优先级（概率越低越先删）
            low_sims = sims_arr[low_sim_indices]
            probs = softmax(low_sims / temperature)  # 注意：这里 similarity 越小，probs 越小
            # 按 probs 升序排序（最不相关在前）
            sorted_order = np.argsort(probs)
            ordered_low_indices = low_sim_indices[sorted_order]
        else:
            ordered_low_indices = np.array([], dtype=int)

        max_remove = int(M * remove_ratio)
        actual_remove_set = set(ordered_low_indices[:max_remove])

        # 确保至少保留 min_keep 个
        if len(actual_remove_set) > M - min_keep:
            # 按优先级保留最后 (M - min_keep) 个
            keep_count = M - min_keep
            actual_remove_set = set(ordered_low_indices[:-keep_count]) if keep_count > 0 else set()

        new_seq = [item for j, item in enumerate(valid_items) if j not in actual_remove_set]
        removed_count = len(actual_remove_set)
        total_removed += removed_count

        new_line = str(uid) + ' ' + ' '.join(map(str, new_seq))
        denoised_lines.append(new_line)

    return denoised_lines, total_removed

def main():
    tmall_dir = os.path.join('dataset', 'Tmall')
    user_emb_dir = os.path.join(tmall_dir, 'User_emb')
    item_emb_dir = os.path.join(tmall_dir, 'Item_emb')

    
    remove_ratio = 0.2          
    similarity_threshold = 0.725
    temperature = 5.0
    min_keep = 1
   

    all_removed = {}

    for filename in ['click.txt', 'cart.txt']:
        filepath = os.path.join(tmall_dir, filename)
        if os.path.exists(filepath):
            denoised, removed = denoise_behavior_file(
                filepath, user_emb_dir, item_emb_dir,
                remove_ratio=remove_ratio,
                similarity_threshold=similarity_threshold,
                temperature=temperature,
                min_keep=min_keep
            )
            output_path = filepath.replace('.txt', '_denoised.txt')
            with open(output_path, 'w') as f:
                for line in denoised:
                    f.write(line + '\n')
            all_removed[filename.replace('.txt', '')] = removed
            print(f"✅ Saved: {output_path} | Removed: {removed}\n")

    # alipay: copy
    alipay_path = os.path.join(tmall_dir, 'alipay.txt')
    if os.path.exists(alipay_path):
        output_path = alipay_path.replace('.txt', '_denoised.txt')
        with open(alipay_path, 'r') as fin, open(output_path, 'w') as fout:
            fout.write(fin.read())
        print(f"✅ Saved: {output_path} (copied unchanged)\n")

    # Summary
    total = sum(all_removed.values())
    print("📊 Denoising Summary:")
    for k, v in all_removed.items():
        print(f"  - {k}.txt: {v} interactions removed")
    print(f"  🎯 Total removed: {total}")
    print(f"  🔒 Threshold: only remove if similarity < {similarity_threshold}")
    print(f"  ⚖️  Max ratio: capped at {remove_ratio*100:.0f}% per user")

if __name__ == "__main__":
    main()