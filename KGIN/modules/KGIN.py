'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


# 模型的核心部分   信息聚合
# 参考这里进行局部模型的聚合   聚合方式稍后再看
# 在这里加指导向量
class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network  基于路径的卷积
    """

    def __init__(self, n_users):  # 不需要传入意图数了
        super(Aggregator, self).__init__()
        self.n_users = n_users
        # self.n_factors = n_factors  # 用户意图数

    def forward(self, entity_emb, item_emb, user_emb, aspect_emb,
                edge_index, edge_type, ua_interact_mat, ia_interact_mat,
                weight):
        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]  # emb_size
        n_users = self.n_users
        # n_factors = self.n_factors

        """KG aggregate  item嵌入：在KG上的"""  # KG消息聚合
        head, tail = edge_index  # 分别取出head和tail对应的下标
        edge_relation_emb = weight[edge_type - 2]  # 因为没有包含u-a,i-a交互关系，所以要-2  源代码：exclude interact, remap [1, n_relations) to [0, n_relations-1)
        # 下面两行对应公式（10）
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]  tail * relation
        # 将同一个head的进行均值计算，也就是（10）  *** 利用这个计算的方式
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities,
                                  dim=0)  # head相同的进行均值运算，也就是公式（10）加和前面的分式

        '''
        """cul user->latent factor attention"""  # 对应公式（8）计算注意力权重 利用这种方式计算注意力
        score_ = torch.mm(user_emb, latent_emb.t())
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]
        '''


        # 计算user对aspect的注意力权重
        score_ua_ = torch.mm(user_emb, aspect_emb.t())  # 得[n_users, n_aspects]
        score_ua = nn.Softmax(dim=1)(score_ua_).unsqueeze(-1)  # [n_users, n_aspects, 1]

        # 计算item对aspect的注意力权重
        score_ia_ = torch.mm(item_emb, aspect_emb.t())  # [n_items, n_aspects]
        score_ia = nn.Softmax(dim=1)(score_ia_).unsqueeze(-1)  # [n_users, n_aspects, 1]


        # 公式(7)
        # print(aspect_emb.shape)  # [11864, 64]
        user_agg = torch.sparse.mm(ua_interact_mat, aspect_emb)  # [n_users, channel]
        user_agg = user_agg * score_ua.sum(dim=1) + user_agg  # [n_users, channel]  在聚合的时候包含了自身  Expected dim 0 size 148363, got 11864


        """item aggregate： 在u-a-i图上的"""
        item_agg = torch.sparse.mm(ia_interact_mat, aspect_emb)  # [n_items, channel]
        item_agg = item_agg * score_ia.sum(dim=1) + item_agg  # [n_items, channel]  在聚合的时候包含了自身

        return item_agg, entity_agg, user_agg  # 按照编号顺序的


# 主要是进行dropout，并调用了Aggregator
# re
class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    # channel：嵌入维度  emb_size
    def __init__(self, channel, n_hops, n_users,
                 n_relations, n_aspects, n_items, ua_interact_mat, ia_interact_mat,  # ind  意图独立性建模用不到
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.ua_interact_mat = ua_interact_mat  # user-item交互邻接矩阵
        self.ia_interact_mat = ia_interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        # self.n_factors = n_factors
        self.n_aspects = n_aspects  # 多加的
        self.n_items = n_items
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = torch.device("cuda:0")
        # self.ind = ind  # 意图独立性建模的方式

        # self.temperature = 0.2  # ？？？？

        # 初始化关系嵌入
        initializer = nn.init.xavier_uniform_  # 初始化方式
        weight = initializer(torch.empty(n_relations - 2, channel)).to(self.device)  # KG中的关系嵌入not include interact  包含除了u-a, i-a交互的关系
        self.weight = nn.Parameter(weight)  # [n_relations - 2, in_channel]  可训练

        # 公式（2）中意图-关系权重矩阵
        # disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))  # 只包含KG中的关系种类，不包括user-item交互关系
        # self.disen_weight_att = nn.Parameter(disen_weight_att)  # trainable

        # 模型   三层聚合
        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        self.W1 = nn.Parameter(torch.ones([1]))
        self.W2 = nn.Parameter(torch.ones([1]))

    # dropout  随机dropout掉一些KG中的边
    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]  转置了
        # edge_type: [-1]
        n_edges = edge_index.shape[1]  # edge_index: [2,5115492]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    # dropout  稀疏矩阵dropout
    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    # def _cul_cor_pro(self):
    #     # disen_T: [num_factor, dimension]
    #     disen_T = self.disen_weight_att.t()
    #
    #     # normalized_disen_T: [num_factor, dimension]
    #     normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)
    #
    #     pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
    #     ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)
    #
    #     pos_scores = torch.exp(pos_scores / self.temperature)
    #     ttl_scores = torch.exp(ttl_scores / self.temperature)
    #
    #     mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
    #     return mi_score

    # 意图独立性建模  3.1.2
    # 意图独立性建模

    def forward(self, user_emb, item_emb, entity_emb, aspect_emb, edge_index, edge_type,
                ua_interact_mat, ia_interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:  # True
            # user-aspect-item-entity
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)  # KG
            ua_interact_mat = self._sparse_dropout(ua_interact_mat, self.node_dropout_rate)
            ia_interact_mat = self._sparse_dropout(ia_interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]  entity和嵌入维度
        user_res_emb = user_emb  # [n_users, channel]   user 和 嵌入维度
        item_res_emb = item_emb
        # cor = self._cul_cor()
        for i in range(len(self.convs)):  # Moudlelist
            item_emb, entity_emb, user_emb = self.convs[i](entity_emb, item_emb, user_emb, aspect_emb,
                                                           edge_index, edge_type, ua_interact_mat, ia_interact_mat,
                                                           self.weight)
            # 这个dropout必须对应吗？？？？
            """message dropout"""
            if mess_dropout:
                item_emb = self.dropout(item_emb)
                user_emb = self.dropout(user_emb)
                entity_emb = self.dropout(entity_emb)
            # 归一化
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""  # 将每一层的嵌入叠加  对应公式（13）*********
            item_res_emb = torch.add(item_res_emb, item_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
            entity_res_emb = torch.add(entity_res_emb, entity_emb)

        item_res_emb = self.W1 * entity_res_emb[:self.n_items] + self.W2 * item_res_emb  # 可以移到上面

        return item_res_emb, user_res_emb  # cor: 意图独立性建模


# main_model
class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, ua_adj_mat, ia_adj_mat):  # 传进来的邻接矩阵是user-item交互矩阵
        super(Recommender, self).__init__()

        # 各个节点的总数
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities + n_apsects
        self.n_aspects = data_config['n_aspects']  # aspect总数

        # 模型超参数
        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        # self.n_factors = args_config.n_factors    # 用户意图数
        # dropout设置
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        # self.ind = args_config.ind   # 意图独立性建模的方式
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.ua_adj_mat = ua_adj_mat  # user-aspect邻接矩阵
        self.ia_adj_mat = ia_adj_mat  # item-aspect邻接矩阵
        self.graph = graph  # KG
        self.edge_index, self.edge_type = self._get_edges(graph)  # KG

        self._init_weight()  # 初始化
        # self.all_embed = nn.Parameter(self.all_embed)  # 所有节点的嵌入：可训练
        self.aspect_emb = nn.Parameter(self.aspect_emb)  # aspect嵌入：可训练
        # self.latent_emb = nn.Parameter(self.latent_emb)  # 意图的嵌入：可训练

        self.gcn = self._init_model()  # 初始化模型
        # 不同item的权重


    # re
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_  # 一种初始化方式，保证输入和输出的方差一致
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size)).to(self.device)  # user 嵌入
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size)).to(self.device)  # item 嵌入  在u-a-i图上的嵌入
        self.entity_emb = initializer(torch.empty(self.n_entities, self.emb_size)).to(self.device)  # entity嵌入：item在KG上的嵌入
        self.aspect_emb = initializer(torch.empty(self.n_aspects, self.emb_size)).to(self.device)  # 每个aspect的嵌入 嵌入维度也是emb_size
        # self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))   # 意图的嵌入

        # [n_users, n_entities] 交互矩阵转化为tensor  不太知道为什么？？？？？
        self.ua_interact_mat = self._convert_sp_mat_to_sp_tensor(self.ua_adj_mat).to(self.device)
        self.ia_interact_mat = self._convert_sp_mat_to_sp_tensor(self.ia_adj_mat).to(self.device)

    # re
    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,  # 消息传播的跳数
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         # n_factors=self.n_factors,
                         n_aspects=self.n_aspects,
                         n_items = self.n_items,
                         ua_interact_mat=self.ua_interact_mat,
                         ia_interact_mat=self.ia_interact_mat,
                         # ind=self.ind,  # 意图独立性建模的方式
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    # 获得边的head、tail 和 id
    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2] 三元组的head和tail
        type = graph_tensor[:, -1]  # [-1, 1]   relation ID
        return index.t().long().to(self.device), type.long().to(self.device)  # 注意index进行了转置

    def forward(self, batch=None):
        user = batch['users']  # 该batch中的user id列表
        pos_item = batch['pos_items']  # 对应的交互过的item的列表
        neg_item = batch['neg_items']  # 每一个正例对应一个反例

        # 初始化的user和item嵌入
        user_emb = self.user_embed
        item_emb = self.item_embed
        entity_emb = self.entity_emb
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        # self.gcn: GraphConv
        item_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                item_emb,
                                                entity_emb,
                                                self.aspect_emb,  # intention的嵌入
                                                self.edge_index,
                                                self.edge_type,
                                                self.ua_interact_mat,
                                                self.ia_interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        # 获得user、pos、neg的嵌入，然后计算相应的指标
        u_e = user_gcn_emb[user]
        # i-a图上的嵌入和
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]

        return self.create_bpr_loss(u_e, pos_e, neg_e)

    # 输入模型计算user和item的表示  看loss值情况判断有没有学到  re
    def generate(self):
        user_emb = self.user_embed
        item_emb = self.item_embed
        entity_emb = self.entity_emb
        return self.gcn(user_emb,
                        item_emb,
                        entity_emb,
                        self.aspect_emb,
                        self.edge_index,
                        self.edge_type,
                        self.ua_interact_mat,
                        self.ia_interact_mat,
                        mess_dropout=False, node_dropout=False)


    # 测试的时候用到了   相乘计算相似度
    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    # BPR损失
    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss, mf_loss, emb_loss
