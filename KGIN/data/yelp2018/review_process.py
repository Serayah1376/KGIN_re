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
2. 得到所有user和item的dict: org_id -> remap_id
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
3. 遍历review，构建user-aspect，item-aspect对应关系
   先得到所有的，之后再区分正负
'''
not_found = 0
with open("u_a.txt", mode='w', encoding='utf-8') as u_a:
    with open("i_a.txt", mode='w', encoding='utf-8') as i_a:
        for review in data:
            u = review['user']
            i = review['item']
            a = review['template'][0]
            s = review['template'][-1]
            aspect1id = aspect_dict[a]
            if u in user:
                userid = user[u]
                u_a.write(str(userid) + " " + str(s) + " " + str(aspect1id) + "\n")
            else:
                not_found += 1

            if i in item:
                itemid = item[i]
                i_a.write(str(itemid) + " " + str(s) + " " + str(aspect1id) + "\n")
            else:
                not_found += 1
