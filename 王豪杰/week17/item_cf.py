import  numpy as np
"""
base item-cf
"""

def dis_jaccard(i1,i2):

    m = len(i1 & i2)
    n = len(i1 | i2)
    if n == 0:
        return 0
    else:
        return np.around(m / n, 2)


def get_item_sim_dict(item_user):

    sim_dict={}
    for iid1 ,uid1_set in item_user.items():
        for iid2, uid2_set in item_user.items():
            if iid1==iid2:
                continue
            else:
                if sim_dict.get(iid1,-1)==-1:
                    sim_dict[iid1]={iid2:dis_jaccard(uid1_set,uid2_set)}
                else:
                    sim_dict[iid1].update({iid2: dis_jaccard(uid1_set, uid2_set)})
    return sim_dict


def predict_iid_score(uid,iid,user_item,sim_dict):

    #1.用户已经点击的序列
    uid_click=user_item[uid]

    #2.遍历用户点击的序列并计算得分综合
    iid_score=0

    for uiid in uid_click:
        iid_score+=sim_dict[iid].get(uiid,0)
    return iid_score/len(uid_click)


def recommand(uid,user_item,sim_dict,all_item_set,top_item=20):

    uid_dict={}
    #拿出用户评价过的物品
    uid_click=user_item[uid]
    uid_unclick=all_item_set-uid_click
    for un_click_id in uid_unclick:
        uid_dict[un_click_id]=predict_iid_score(uid,un_click_id,user_item,sim_dict)

    re=dict(sorted(uid_dict.items(), key=lambda x: x[1], reverse=True)[:top_item])
    return re


def create_item_user_click(path):

    #初始化用户物品字典为空
    item_user = dict()
    user_item=dict()
    all_item_set=set()
    #相当于打开文件操作，做一个buffer
    with open(path, "r", encoding="utf-8") as f:
        #死循环，一行一行读取数据，知道读取完毕
        while True:
            #一行一行读数据 1	1	5	874965758
            line = f.readline()
            # 如果line不为空，则对line基于\t进行切分，得到[1,1,5,874965758]
            if line:
                lines = line.strip().split("\t")
                uid = lines[0]
                iid = lines[1]
                all_item_set.add(iid)
                # 初始化字典,get到uid就更新 如果uid不在字典中，那么初始化uid为
                #key，value为set(iid)
                if item_user.get(iid, -1) == -1:
                    item_user[iid] ={uid}
                else:
                    item_user[iid].add(uid)

                if user_item.get(uid, -1) == -1:
                    user_item[uid] = {iid}
                else:
                    user_item[uid].add(iid)

            #如果line为空，表示读取完毕，那么调出死循环。
            else:
                print("读完")
                break
    return item_user,user_item,all_item_set


if __name__ == '__main__':

    item_user,user_item,all_item_set=create_item_user_click(r"D:\badou_nlp\第十七周 推荐系统\week17 推荐系统\ml-100k\u.data")

    #相似矩阵

    sim_dict=get_item_sim_dict(item_user)

    s=predict_iid_score("1","2",user_item,sim_dict)

    re=recommand("2", user_item, sim_dict, all_item_set, top_item=20)
    print(len(re))
