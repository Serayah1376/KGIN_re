'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

# 可以跑通，作为baseline

import random

import torch
import numpy as np

from time import time
from prettytable import PrettyTable

from utils_KGIN.parser import parse_args
from utils_KGIN.data_loader import load_data
from modules.KGIN import Recommender
from utils_KGIN.evaluate import test
from utils_KGIN.helper import early_stopping

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

# 获得pos和neg  为计算loss和训练做准备
def get_feed_dict(train_cf_pairs, start, end, train_user_set):
    # 每一个正例，会对应一个反例
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_cf_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]  # user id列表
    feed_dict['pos_items'] = entity_pairs[:, 1]  # item id列表  是从零开始的   说明item的嵌入在前面
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020   # 改成2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""  # 参数和设备
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf_ua, train_cf_ia, train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)  # 已改
    adj_mat_list, mean_mat_list = mat_list  # 邻接矩阵 包含u-a, i-a

    n_users = n_params['n_users']  # user总数
    n_items = n_params['n_items']  # item总数
    n_entities = n_params['n_entities']   # entity总数
    n_relations = n_params['n_relations']   # 关系数量
    n_nodes = n_params['n_nodes']   # 节点总数量  n_entities + n_users
    n_aspects = n_params['n_aspects']  # aspect总数

    """cf data"""  # 将user-item对转化为tensor [[userid,aspectid]]、[[itemid,aspectid]]
    # train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    train_cf_ua = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf_ua], np.int32))
    train_cf_ia = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf_ia], np.int32))
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0], mean_mat_list[1]).to(device)    # 0维：user-aspect  1维：item-aspect

    """define optimizer"""   # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    # 暂停的步数
    stopping_step = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):
        """training CF（u-a，i-a）"""
        '''
        # shuffle training data
        index = np.arange(len(train_cf))  # 训练集
        np.random.shuffle(index)  # 打乱
        train_cf_pairs = train_cf_pairs[index] # 打乱后的训练集
        '''

        """training"""
        loss, s = 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf_ua):
            # 生成反面例子
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'])
            batch_loss, _, _ = model(batch)

            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            # cor_loss += batch_cor
            s += args.batch_size

        train_e_t = time()

        # 测试  每10个epoch测试一次
        if epoch % 10 == 9 or epoch == 1:
            """testing"""
            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
            )
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            # cur_best_pre_0: 记录最好的结果
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)

            if should_stop:
                # print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))

