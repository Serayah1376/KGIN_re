'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

# 传进来的log_value是recall@20
def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):  # 传进来的early_stop是10
    # early sto3pping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):  # 两种指标，一个大了好，另一个小了好
        stopping_step = 0
        best_value = log_value
    else:  # 如果啥改变的话就停止
        stopping_step += 1   # 尚未运行

    # print(flag_step)

    if stopping_step >= flag_step:   # 太久没有提升
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))  # 没有这句话
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
