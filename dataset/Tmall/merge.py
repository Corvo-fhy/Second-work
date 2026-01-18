from collections import defaultdict

def read_file_with_order(filename):
    """读取文件，返回 user->items 映射 + 用户出现顺序"""
    user_items = {}
    user_order = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user = parts[0]
            items = set(parts[1:])
            if user not in user_items:  # 避免重复用户（通常不会）
                user_order.append(user)
            user_items[user] = items
    return user_items, user_order

def write_file_preserve_order(filename, user_items, user_order):
    """按指定用户顺序写入文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        for user in user_order:
            if user in user_items and user_items[user]:  # 跳过空行为（可选）
                items = sorted(user_items[user], key=int)
                f.write(user + ' ' + ' '.join(items) + '\n')

def insert_before_ext(filename, suffix='_fused'):
    """在 .txt 前插入后缀，如 'a.txt' -> 'a_fused.txt'"""
    if filename.endswith('.txt'):
        return filename[:-4] + suffix + '.txt'
    else:
        return filename + suffix  # fallback

# === 主逻辑 ===
cart_filename = 'cart_ori_noisy.txt'
click_filename = 'click_ori_noisy.txt'

# 读取文件，保留顺序
train, train_order = read_file_with_order('train.txt')
cart, cart_order = read_file_with_order(cart_filename)
click, click_order = read_file_with_order(click_filename)

# 确定最终用户顺序：
# 优先按 train 的顺序，然后补充 cart 中不在 train 的用户（按 cart 顺序），
# 再补充 click 中前两者都没有的用户（按 click 顺序）
final_user_order = []
seen = set()

for user in train_order:
    if user not in seen:
        final_user_order.append(user)
        seen.add(user)

for user in cart_order:
    if user not in seen:
        final_user_order.append(user)
        seen.add(user)

for user in click_order:
    if user not in seen:
        final_user_order.append(user)
        seen.add(user)

# 构建融合后的行为
new_cart = {}
new_click = {}

for user in final_user_order:
    t = train.get(user, set())
    c = cart.get(user, set())
    k = click.get(user, set())
    
    # 强制层级关系: train ⊆ cart ⊆ click
    fused_cart = c | t
    fused_click = k | fused_cart
    
    new_cart[user] = fused_cart
    new_click[user] = fused_click

# 生成正确文件名
cart_output = insert_before_ext(cart_filename, '_fused')
click_output = insert_before_ext(click_filename, '_fused')

# 写回文件（保持顺序）
write_file_preserve_order(cart_output, new_cart, final_user_order)
write_file_preserve_order(click_output, new_click, final_user_order)

print(f"融合完成！")
print(f"输出文件: {cart_output}, {click_output}")