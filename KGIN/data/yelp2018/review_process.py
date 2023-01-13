import pickle

file = open('reviews.pickle', 'rb')
data = pickle.load(file)  # 读取存储的pickle文件  list类型

'''
1. 得到所有aspect的编码
'''
# 将所有的aspect都放在一个set中然后进行编码
aspect = list()
for a in data:
    aspect.append(a['template'][0])

# 先转化为set，再转换成dict
aspect_set = set(aspect)
aspect_dict = dict()
for a, b in enumerate(aspect_set):
    aspect_dict[b] = a

'''
2. 得到所有user和item的dict: org_id -> remap_id 为了将review与原数据集链接起来
'''
# 将user_list.txt转换成字典
with open('user_list.txt', 'r', encoding='utf-8')as f:
    user = dict()
    head = True
    for line in f.readlines():
        if head:
            head = False
            continue
        line = line.strip('\n')
        a = line.split(' ')
        user[a[0]] = a[1]

# 将item_list.txt转换成字典
with open('entity_list.txt', 'r', encoding='utf-8')as f:
    item = dict()
    head = True
    for line in f.readlines():
        if head:
            head = False
            continue
        line = line.strip('\n')
        b = line.split(' ')
        item[b[0]] = b[1]

'''
    3. 生成user-aspect和item-aspect的训练集和测试集
    其中test只需要u-i交互，因为有aspect就说明有交互
'''
train_f = open('1/train.txt', mode='r', encoding='utf-8')
train = train_f.read()
train = train.split(' ')
train = list(map(int, train))  # 训练集索引列表, type：list

test_f = open('1/test.txt', mode='r', encoding='utf-8')
test = test_f.read()
test = test.split(' ')
test = list(map(int, test))  # 训练集索引列表, type：list

# 先写u_a_train.txt
count = 0
not_count = 0
with open("i_a_train.txt", mode='w', encoding='utf-8') as i_a_train:
    with open("u_a_train.txt", mode='w', encoding='utf-8') as u_a_train:
        with open("test_re.txt", mode='w', encoding='utf-8') as test_re:
            with open("train_re.txt", mode='w', encoding='utf-8') as train_re:
                for review in data:  # 与索引对应
                    u = review['user']
                    i = review['item']
                    a = review['template'][0]
                    s = review['template'][-1]
                    aspectid = aspect_dict[a]
                    if u in user.keys() and i in item.keys() :
                        userid = user[u]
                        itemid = item[i]
                        if int(itemid) > 45537:  # 大于的筛掉
                            continue
                        if count in train:   # 属于训练集
                            u_a_train.write(str(userid) + " " + str(s) + " " + str(aspectid) + "\n")
                            i_a_train.write(str(itemid) + " " + str(s) + " " + str(aspectid) + "\n")
                            train_re.write(str(userid) + " " + str(itemid) + "\n")
                        else:
                            test_re.write(str(userid) + " " + str(itemid) + "\n")
                    else:
                        not_count += 1
                    count += 1
                    print(count)

print("not_count: ", not_count)




