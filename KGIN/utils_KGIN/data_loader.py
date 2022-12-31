import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

'''
采用KB4Rec的方法将Amazon-book和Last-FM数据集通过名称匹配对应到Freebase
'''

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


# 读交互数据
# 得到[userID,itemID]二元组对
def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))  # 去重并转换成list
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)

# 重新对应训练集和测试集中的userID和itemID, 变成{userid，[item1ID,item2ID,item3ID,......]}
def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1   # user的数量
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1   # item的数量

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


# 读取KG，并考虑反向关系三元组，返回所有三元组
def read_triplets(file_name):   # kg_final.txt
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)  # 去掉重复的三元组

    if args.inverse_r:   # 是否考虑反向关系  默认为True
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        # 交换前后两个的entity的关系*****
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1   # 新的关系id
        # consider two additional relations --- 'interact' and 'be interacted'
        # 考虑交互关系，所有关系id+1
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets

# KG  和 {reletionID,[h_id/u_id,t_id/i_id]}
def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()  # 有向图
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    # 交互关系
    for u_id, i_id in tqdm(train_data, ascii=True):  # 传入可迭代对象
        rd[0].append([u_id, i_id])  # 交互关系的id是0

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)   # 添加一条边，并标记relation的ID
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd  # 知识图谱、关系与entity的对应

# 返回原始邻接矩阵，D^{-1/2}AD^{-1/2}归一化的邻接矩阵，D^{-1}A归一化的邻接矩阵
def build_sparse_relational_graph(relation_dict):
    # 两种不同的归一化方法
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))  # 度

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()  # 转换为稀疏矩阵

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")  # 稀疏关系矩阵
    # 每种关系建立一个邻接矩阵
    for r_id in tqdm(relation_dict.keys()):  # 遍历所有的relation
        np_mat = np.array(relation_dict[r_id])  # 将同一关系的[head,tail]转换成array
        if r_id == 0:  # user-item 交互关系
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)  调整ID
            vals = [1.] * len(cf)  # 填充的值都为1
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))  # 交互邻接矩阵
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)  # 每一种关系维护一个邻接矩阵

    # 逐个进行两种方式的归一化
    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list   # 后两者归一化的方式不同


# 主函数 数据加载
def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'   # 数据集路径

    print('reading train and test user-item set ...')
    # 读训练集和测试集交互数据
    train_cf = read_cf(directory + 'train.txt')  # 得到[userID,itemID]二元组对
    test_cf = read_cf(directory + 'test.txt')   # 得到[userID,itemID]二元组对
    remap_item(train_cf, test_cf)  # 重新对应训练集和测试集中的userID和itemID, 变成{userid，[item1ID,item2ID,item3ID,......]}

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + 'kg_final.txt')   # 得到KG三元组，以及关系反向后的三元组

    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)  # graph: KG， relation_dict: 关系字典[relation;[head,tail]]

    print('building the adj mat ...')   # 邻接矩阵  稀疏
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph, \
           [adj_mat_list, norm_mat_list, mean_mat_list]

