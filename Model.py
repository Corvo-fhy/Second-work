import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
import copy
from utility.helper import *
from utility.batch_test import *
import multiprocessing
import torch.multiprocessing
import random
import pickle
from torch_scatter import scatter_softmax, scatter_add
from sklearn.metrics.pairwise import cosine_similarity


def load_text_embeddings_separate(user_dir, item_dir):
    user_emb_dict = {}
    item_emb_dict = {}

    for fname in os.listdir(user_dir):
        if fname.endswith('.npy'):
            uid = int(os.path.splitext(fname)[0])
            emb = np.load(os.path.join(user_dir, fname))
            user_emb_dict[uid] = torch.tensor(emb, dtype=torch.float32)

    for fname in os.listdir(item_dir):
        if fname.endswith('.npy'):
            iid = int(os.path.splitext(fname)[0])
            emb = np.load(os.path.join(item_dir, fname))
            item_emb_dict[iid] = torch.tensor(emb, dtype=torch.float32)

    return user_emb_dict, item_emb_dict


def dict_to_aligned_tensor(embedding_dict, expected_length):
    emb_dim = next(iter(embedding_dict.values())).shape[0]
    result = torch.zeros((expected_length, emb_dim), dtype=torch.float32)
    for k, v in embedding_dict.items():
        if 0 <= k < expected_length:
            result[k] = v
        else:
            print(f"[Warning] Skipping out-of-bound key: {k}")
    return result

# ========================= 多头邻居限定缩放点积注意力 ========================= #
class RelationMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, rel_dim, num_users):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.num_users = num_users

        # QKV 投影
        self.linear_q = nn.Linear(in_dim, out_dim)
        self.linear_k = nn.Linear(in_dim, out_dim)
        self.linear_v = nn.Linear(in_dim, out_dim)

        # 输出投影
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj, relation_emb, user_text_embs, item_text_embs):
        N = x.size(0)
        device = x.device

        # =================== 融合文本信息 =================== #
        text_emb = torch.zeros_like(x)
        text_emb[:self.num_users] = user_text_embs.to(device)
        text_emb[self.num_users:] = item_text_embs.to(device)

        # =================== QKV =================== #
        Q = self.linear_q(text_emb).view(N, self.num_heads, self.d_k)
        K = self.linear_k(text_emb).view(N, self.num_heads, self.d_k)
        V = self.linear_v(x).view(N, self.num_heads, self.d_k)

        # 邻居限定
        row, col = adj._indices()
        Q_ = Q[col]
        K_ = K[row]
        V_ = V[row]

        # relation embedding
        rel = relation_emb.view(1, self.num_heads, self.d_k).expand(Q_.size(0), -1, -1)

        # =================== 缩放点积注意力 =================== #
        scores = (Q_ * (K_ + rel)).sum(dim=-1) / (self.d_k ** 0.5)
        attn = scatter_softmax(scores, col.unsqueeze(1).expand(-1, self.num_heads), dim=0)
        attn = attn.unsqueeze(-1)
        attn_V = attn * V_

        out = scatter_add(attn_V, col.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.d_k),
                          dim=0, dim_size=N)
        out = out.reshape(N, -1)
        return self.out_proj(out)


# ========================= GNN 主模型 ========================= #
class MyModel(nn.Module):
    def __init__(self, max_item_list, data_config, args, text_dim=768, mlp_dim=64):
        super().__init__()
        self.max_item_list = max_item_list
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.num_nodes = self.n_users + self.n_items
        self.pre_adjs = data_config['pre_adjs']
        self.pre_aug = data_config['pre_aug']
        self.pre_adjs_tensor = [self._convert_sp_mat_to_sp_tensor(adj).to(device) for adj in self.pre_adjs]
        self.aug_mat = self._convert_sp_mat_to_sp_tensor(self.pre_aug).to(device)
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)

        self.coefficient = torch.tensor(eval(args.coefficient)).view(1, -1).to(device)
        self.emb_dim = args.embed_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.mess_dropout = eval(args.mess_dropout)
        self.nhead = args.nhead

        # =================== GNN embedding =================== #
        self.all_weights = nn.ParameterDict({
            'user_embedding': Parameter(torch.FloatTensor(self.n_users, self.emb_dim)),
            'item_embedding': Parameter(torch.FloatTensor(self.n_items, self.emb_dim)),
            'relation_embedding': Parameter(torch.FloatTensor(self.n_relations, self.emb_dim)),
        })
        self.weight_size_list = [self.emb_dim] + self.weight_size
        for k in range(self.n_layers):
            self.all_weights[f'W_gc_{k}'] = Parameter(torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k+1]))
            self.all_weights[f'W_rel_{k}'] = Parameter(torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k+1]))

        # =================== 文本 embedding =================== #
        text_dim = 768
        emb_dim = self.emb_dim  # 图嵌入维度，比如 64

        self.mlp_user = nn.Sequential(
            nn.Linear(text_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.mlp_item = nn.Sequential(
            nn.Linear(text_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
)


        # =================== Attention layers =================== #
        self.relation_attn_layers = nn.ModuleList([
            RelationMultiHeadAttention(
                in_dim=self.emb_dim,
                out_dim=self.emb_dim,
                num_heads=self.nhead,
                rel_dim=self.emb_dim,
                num_users=self.n_users
            ) for _ in range(self.n_relations)
        ])

        self.reset_parameters()
        self.dropout = nn.Dropout(self.mess_dropout[0])
        self.leaky_relu = nn.LeakyReLU()

    def set_text_embeddings(self, user_text_tensor, item_text_tensor):
        self.text_emb_user = user_text_tensor
        self.text_emb_item = item_text_tensor

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.all_weights['user_embedding'])
        nn.init.xavier_uniform_(self.all_weights['item_embedding'])
        nn.init.xavier_uniform_(self.all_weights['relation_embedding'])
        for k in range(self.n_layers):
            nn.init.xavier_uniform_(self.all_weights[f'W_gc_{k}'])
            nn.init.xavier_uniform_(self.all_weights[f'W_rel_{k}'])

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape
        return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape))

    # =================== Forward =================== #
    def forward(self, device):
        assert self.text_emb_user is not None and self.text_emb_item is not None
        # MLP降维文本 embedding
        h_user = self.mlp_user(self.text_emb_user.to(device))
        h_item = self.mlp_item(self.text_emb_item.to(device))

        ego_embeddings = torch.cat((self.all_weights['user_embedding'], self.all_weights['item_embedding']), dim=0)
        ego_aug_embeddings = ego_embeddings

        all_rela_embs = {beh: [self.all_weights['relation_embedding'][i].unsqueeze(0)] 
                         for i, beh in enumerate(self.behs)}

        all_embeddings = [ego_embeddings]
        all_embeddings_aug = [ego_aug_embeddings]

        for k in range(self.n_layers):
            embeddings_list = []
            for i, beh in enumerate(self.behs):
                rela_emb = all_rela_embs[beh][k]
                embeddings_ = self.relation_attn_layers[i](
                    ego_embeddings,
                    self.pre_adjs_tensor[i],
                    rela_emb,
                    h_user,
                    h_item
                )
                embeddings_ = self.leaky_relu(torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights[f'W_gc_{k}']))
                embeddings_list.append(embeddings_)

            embeddings_st = torch.stack(embeddings_list, dim=1)
            embeddings = sum(embeddings_st[:, idx, :] * self.coefficient[0, idx] for idx in range(self.n_relations))
            embeddings = self.dropout(embeddings)
            all_embeddings.append(embeddings)

            # aug 图
            embeddings_list_aug = []
            for i, beh in enumerate(self.behs):
                rela_emb = all_rela_embs[beh][k]
                mat = self.pre_adjs_tensor[i] if i != self.n_relations - 1 else self.aug_mat
                x_input = ego_embeddings if i != self.n_relations - 1 else ego_aug_embeddings
                embeddings_ = self.relation_attn_layers[i](x_input, mat, rela_emb, h_user, h_item)
                embeddings_ = self.leaky_relu(torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights[f'W_gc_{k}']))
                embeddings_list_aug.append(embeddings_)

            embeddings_st = torch.stack(embeddings_list_aug, dim=1)
            embeddings = sum(embeddings_st[:, idx, :] * self.coefficient[0, idx] for idx in range(self.n_relations))
            embeddings = self.dropout(embeddings)
            all_embeddings_aug.append(embeddings)

            # 更新关系 embedding
            for i, beh in enumerate(self.behs):
                rela_emb = torch.matmul(all_rela_embs[beh][k], self.all_weights[f'W_rel_{k}'])
                all_rela_embs[beh].append(rela_emb)

        all_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        all_embeddings_aug = torch.mean(torch.stack(all_embeddings_aug, dim=1), dim=1)

        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        token_embedding = torch.zeros([1, self.emb_dim], device=device)
        i_g_embeddings = torch.cat((i_g_embeddings, token_embedding), dim=0)

        u_g_embeddings_aug, i_g_embeddings_aug = torch.split(all_embeddings_aug, [self.n_users, self.n_items], 0)
        i_g_embeddings_aug = torch.cat((i_g_embeddings_aug, token_embedding), dim=0)

        for beh in self.behs:
            all_rela_embs[beh] = torch.mean(torch.stack(all_rela_embs[beh], 0), dim=0)

        return u_g_embeddings, i_g_embeddings, all_rela_embs, u_g_embeddings_aug, i_g_embeddings_aug, h_user, h_item

# ========================= 对齐 Loss ========================= #
def alignment_loss(gnn_emb, mlp_emb):
    gnn_emb = F.normalize(gnn_emb, dim=1)
    mlp_emb = F.normalize(mlp_emb, dim=1)
    return 1 - (gnn_emb * mlp_emb).sum(dim=1).mean()


class RecLoss(nn.Module):
    def __init__(self, data_config, args):
        super(RecLoss, self).__init__()
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_dim = args.embed_size
        self.coefficient = eval(args.coefficient)
        self.wid = eval(args.wid)

    def forward(self, input_u, label_phs, ua_embeddings, ia_embeddings, rela_embeddings):
        uid = ua_embeddings[input_u.long()]
        uid = uid.squeeze()

        pos_r_list = []
        for i in range(self.n_relations):
            beh = self.behs[i]
            pos_beh = ia_embeddings[label_phs[i].long()]  # [B, max_item, dim]
            pos_num_beh = torch.ne(label_phs[i], self.n_items).float()
            pos_beh = torch.einsum('ab,abc->abc', pos_num_beh,
                                   pos_beh)  # [B, max_item] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ac,abc->abc', uid,
                                 pos_beh)  # [B, dim] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ajk,lk->aj', pos_r, rela_embeddings[beh])
            pos_r_list.append(pos_r)

        loss = 0.
        for i in range(self.n_relations):
            beh = self.behs[i]
            # 后半部分
            temp = torch.einsum('ab,ac->bc', ia_embeddings, ia_embeddings) \
                   * torch.einsum('ab,ac->bc', uid, uid)  # [B, dim]' * [B, dim] -> [dim, dim]
            tmp_loss = self.wid[i] * torch.sum(
                temp * torch.matmul(rela_embeddings[beh].T, rela_embeddings[beh]))
            tmp_loss += torch.sum((1.0 - self.wid[i]) * torch.square(pos_r_list[i]) - 2.0 * pos_r_list[i])

            loss += self.coefficient[i] * tmp_loss

        regularizer = torch.sum(torch.square(uid)) * 0.5 + torch.sum(torch.square(ia_embeddings)) * 0.5
        emb_loss = args.decay * regularizer

        return loss * 100, emb_loss


def get_lables(temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()
    max_item = item_lenth[int(len(item_lenth) * k) - 1]

    print(max_item)
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances1(max_item_list, beh_label_list):
    user_train = []
    beh_item_list = [list() for i in range(n_behs)]  #

    for i in beh_label_list[-1].keys():
        user_train.append(i)
        beh_item_list[-1].append(beh_label_list[-1][i])
        for j in range(n_behs - 1):
            if not i in beh_label_list[j].keys():
                beh_item_list[j].append([n_items] * max_item_list[j])
            else:
                beh_item_list[j].append(beh_label_list[j][i])

    user_train = np.array(user_train)
    beh_item_list = [np.array(beh_item) for beh_item in beh_item_list]
    user_train = user_train[:, np.newaxis]
    return user_train, beh_item_list


def get_train_pairs(user_train_batch, beh_item_tgt_batch):
    input_u_list, input_i_list = [], []
    for i in range(len(user_train_batch)):
        pos_items = beh_item_tgt_batch[i][np.where(beh_item_tgt_batch[i] != n_items)]  # ndarray [x,]
        uid = user_train_batch[i][0]
        input_u_list += [uid] * len(pos_items)
        input_i_list += pos_items.tolist()

    return np.array(input_u_list).reshape([-1]), np.array(input_i_list).reshape([-1])


def test_torch(ua_embeddings, ia_embeddings, rela_embedding, users_to_test, batch_test_flag=False):
    def get_score_np(ua_embeddings, ia_embeddings, rela_embedding, users, items):
        ug_embeddings = ua_embeddings[users]  # 提取用户嵌入
        pos_ig_embeddings = ia_embeddings[items]  # 提取物品嵌入
        
        dot = np.multiply(pos_ig_embeddings, rela_embedding)  # 物品嵌入 * 关系嵌入
        batch_ratings = np.matmul(ug_embeddings, dot.T)  # [U, dim] * [dim, I] -> [U, I]
        return batch_ratings


    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    test_users = users_to_test
    n_test_users = len(test_users)

    # pool = torch.multiprocessing.Pool(cores)
    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        rate_batch = get_score_np(ua_embeddings, ia_embeddings, rela_embedding, user_batch, item_batch)

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users
    assert count == n_test_users

    pool.close()
    return result


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def preprocess_sim(config):
    # 读取相似的用户与物品
    user = sp.load_npz(f"Sim/{config['dataset']}_user_matrix.npz")
    user_indices = torch.tensor(user.toarray()).bool()

    item = sp.load_npz(f"Sim/{config['dataset']}_item_matrix.npz")
    item_indices = torch.tensor(item.toarray()).bool()

    return user_indices, item_indices


def softmax(x, tau=1.0):
    """数值稳定的 Softmax"""
    x = np.array(x)
    x_max = np.max(x)
    exp_x = np.exp((x - x_max) / tau)
    return exp_x / (np.sum(exp_x) + 1e-12)


def filter_by_softmax_similarity(
    user_train,
    beh_item_list,
    user_modal_emb,
    item_modal_emb,
    n_items,
    softmax_threshold=0.05,   # Softmax 概率低于此值则删除
    temperature=1.0,          # Softmax 温度参数
    n_behs=None
):
    if n_behs is None:
        n_behs = len(beh_item_list)
    
    # 转为 NumPy
    if isinstance(user_modal_emb, torch.Tensor):
        user_modal_emb = user_modal_emb.cpu().numpy()
    if isinstance(item_modal_emb, torch.Tensor):
        item_modal_emb = item_modal_emb.cpu().numpy()

    user_ids = user_train.flatten()
    N = len(user_ids)

    for j in range(n_behs - 1):  # 非目标行为
        interactions = beh_item_list[j].copy()
        L = interactions.shape[1]

        for i in range(N):
            uid = user_ids[i]
            items = interactions[i]
            valid_mask = (items != n_items)
            if not np.any(valid_mask):
                continue

            valid_items = items[valid_mask]
            M = len(valid_items)

            # 计算原始相似度
            u_emb = user_modal_emb[uid].reshape(1, -1)
            v_embs = item_modal_emb[valid_items]
            sims_raw = cosine_similarity(u_emb, v_embs).flatten()  # (M,)

            # Softmax 归一化
            probs = softmax(sims_raw, tau=temperature)  # (M,)

            # 删除概率低于阈值的交互
            to_remove = probs < softmax_threshold
            valid_items[to_remove] = n_items

            # 写回
            interactions[i][valid_mask] = valid_items

        beh_item_list[j] = interactions

    return beh_item_list


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.set_device(1)
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(2020)

    config = dict()
    config['device'] = device
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['behs'] = data_generator.behs
    config['trn_mat'] = data_generator.trnMats[-1]  # 目标行为交互矩阵
    
    data_info = [
                "wid : %s" % (args.wid),
                 "decay: %0.2f" % (args.decay),
                 "coefficient : %s" % (args.coefficient)
    ]
    data_info = "\n".join(data_info)
    print(data_info)


    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """

    # 每个行为的预处理后的邻接矩阵
    # D^-0.5 * A * D^-0.5
    pre_adj_list, pre_adj_aug = data_generator.get_adj_mat()
    config['pre_adjs'] = pre_adj_list
    config['pre_aug'] = pre_adj_aug
    print('use the pre adjcency matrix')
    n_users, n_items = data_generator.n_users, data_generator.n_items
    behs = data_generator.behs
    n_behs = data_generator.beh_num

    # user_indices, item_indices = preprocess_sim(config)
    trnDicts = copy.deepcopy(data_generator.trnDicts)
    max_item_list = []
    beh_label_list = []

    # 给用户的交互进行统一化
    # 将用户的交互个数填充为固定值，掩码为n_item
    # 共有7977个item，编号为0~7976，利用7977作为掩码填充
    # beh_label_list存放填充之后的交互信息
    for i in range(n_behs):
        max_item, beh_label = get_lables(trnDicts[i])
        max_item_list.append(max_item)
        beh_label_list.append(beh_label)

    t0 = time()

    # 初始化模型
    model = MyModel(max_item_list, data_config=config, args=args).to(device)
    recloss = RecLoss(data_config=config, args=args).to(device)
    
    user_dir = "dataset/Tmall/User_emb"
    item_dir = "dataset/Tmall/Item_emb"
    user_emb_dict, item_emb_dict = load_text_embeddings_separate(user_dir, item_dir)

    user_text_tensor = dict_to_aligned_tensor(user_emb_dict, n_users).to(device)
    item_text_tensor = dict_to_aligned_tensor(item_emb_dict, n_items).to(device)

    print("user_text_embs.shape:", user_text_tensor.shape)
    print("self.num_users:", n_users)


    model.set_text_embeddings(user_text_tensor, item_text_tensor)


    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 动态调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_gamma)
    cur_best_pre_0 = 0.
    print('without pretraining.')

    run_time = 1

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False

    # user_train1为用户的ID序列，形状为（n_users,1）ndarry
    # beh_item_list为一个List，List的元素个数为n_beh
    # List中的每个元素为一个ndarry，就是将用户的交互转化为ndarry，形状为(n_users, max_item)
    user_train1, beh_item_list = get_train_instances1(max_item_list, beh_label_list)

    nonshared_idx = -1

    sim_threshold = 0.2  # 硬噪声筛选阈值


    beh_item_list = filter_by_softmax_similarity(
    user_train=user_train1,
    beh_item_list=beh_item_list,
    user_modal_emb=user_text_tensor,
    item_modal_emb=item_text_tensor,
    n_items=n_items,
    softmax_threshold=sim_threshold,   
    temperature=1.0
    )

    for epoch in range(args.epoch):
        # 在每个epoch中将数据的顺序打乱
        model.train()
        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        beh_item_list = [beh_item[shuffle_indices] for beh_item in beh_item_list]

        t1 = time()
        loss, rec_loss, emb_loss, ssl_loss, align_loss = 0., 0., 0., 0., 0.
        n_batch = int(len(user_train1) / args.batch_size)

        for idx in range(n_batch):
            optimizer.zero_grad()

            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            beh_batch = [beh_item[start_index:end_index] for beh_item in beh_item_list]

            u_batch_list, i_batch_list = get_train_pairs(user_train_batch=u_batch,
                                                        beh_item_tgt_batch=beh_batch[-1])

            # load into cuda
            u_batch = torch.from_numpy(u_batch).to(device)
            beh_batch = [torch.from_numpy(beh_item).to(device) for beh_item in beh_batch]
            u_batch_list = torch.from_numpy(u_batch_list).to(device)
            i_batch_list = torch.from_numpy(i_batch_list).to(device)

            # ------------------- forward -------------------
            ua_embeddings, ia_embeddings, rela_embeddings, ua_embeddings_aug, ia_embeddings_aug, h_user, h_item = model(device)

            # ------------------- SSL loss -------------------
            ssl_temp = 0.1
            user_emb1 = ua_embeddings[u_batch_list]
            user_emb2 = ua_embeddings_aug[u_batch_list]
            normalize_user_emb1 = F.normalize(user_emb1, dim=1)
            normalize_user_emb2 = F.normalize(user_emb2, dim=1)
            normalize_all_user_emb2 = F.normalize(ua_embeddings_aug, dim=1)

            pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2), dim=1)
            pos_score_user = torch.exp(pos_score_user / ssl_temp)
            ttl_score_user = torch.matmul(normalize_user_emb1, normalize_all_user_emb2.T)
            ttl_score_user = torch.sum(torch.exp(ttl_score_user / ssl_temp), dim=1)
            batch_inter_ssl_loss = -torch.sum(torch.log(pos_score_user / ttl_score_user)) * 0.1

            # ------------------- MLP 模态对齐 loss -------------------
            u_batch_emb = ua_embeddings[u_batch_list]   # [B, emb_dim]
            i_batch_emb = ia_embeddings[i_batch_list]   # [B, emb_dim]

            user_align_loss = 1 - F.cosine_similarity(u_batch_emb, h_user[u_batch_list], dim=1).mean()
            item_align_loss = 1 - F.cosine_similarity(i_batch_emb, h_item[i_batch_list], dim=1).mean()

            batch_align_loss = (user_align_loss + item_align_loss) * 10

            # ------------------- Rec/Emb loss -------------------
            batch_rec_loss, batch_emb_loss = recloss(u_batch, beh_batch, ua_embeddings, ia_embeddings, rela_embeddings)
            # ------------------- 总 loss -------------------
            batch_loss = batch_rec_loss + batch_emb_loss + batch_inter_ssl_loss + batch_align_loss
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item() / n_batch
            rec_loss += batch_rec_loss.item() / n_batch
            emb_loss += batch_emb_loss.item() / n_batch
            ssl_loss += batch_inter_ssl_loss.item() / n_batch
            align_loss += batch_align_loss.item() / n_batch

        if args.lr_decay: scheduler.step()
        torch.cuda.empty_cache()

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % args.test_epoch != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, rec_loss, emb_loss, ssl_loss, align_loss)
                print(perf_str)
            continue

        t2 = time()
        model.eval()
        # 测试模型
        with torch.no_grad():
            ua_embeddings, ia_embeddings, rela_embeddings, _, _, _, _ = model(device)
            users_to_test = list(data_generator.test_set.keys())

            ret = test_torch(ua_embeddings.detach().cpu().numpy(),
                             ia_embeddings.detach().cpu().numpy(),
                             rela_embeddings[behs[-1]].detach().cpu().numpy(), users_to_test)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]:, recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (
                           epoch, t2 - t1, t3 - t2, ret['recall'][1],
                           ret['recall'][3],
                           ret['precision'][1], ret['precision'][3], ret['hit_ratio'][1], ret['hit_ratio'][3],
                           ret['ndcg'][1], ret['ndcg'][3])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop, flag = early_stopping_new(ret['recall'][1], cur_best_pre_0,
                                                                              stopping_step, expected_order='acc',
                                                                              flag_step=10)
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.4f' % r for r in recs[idx]]),
                  '\t'.join(['%.4f' % r for r in ndcgs[idx]]))
    print(final_perf)
    save_dict = f"{data_generator.dataset_name}.pth.tar"
    torch.save(model.state_dict(), save_dict)
    print(f"Model Save To {save_dict}")