"""
这个类是为了实现bpe算法以及硬encoding部分和decoding部分
"""
from collections import defaultdict
import os


class MyDiyBpe:
    def __init__(self,path,max_vocab_size):
        # 这个值代表我所指定的最大的字表大小
        self.max_vocab_size=max_vocab_size
        # 这里面放的是解码之后合并的码值的对应值
        self.merge_drct=defaultdict(dict)
        # 这里面放的是解码之后全部文件中的码值地址
        self.all_dict=defaultdict(dict)
        # 这里面放的是解码之后每个文件中的码值，用来和最新的做对比
        self.origin_elem_of_dict=defaultdict(dict)
        #z 这里面放的是合并码值之后的码值
        self.new_elem_of_dict=defaultdict(dict)
        # 这里面放的是原始的文本信息，作为比对
        self.origin_sentence=defaultdict(dict)
        # 这里直接开始初始化的时候就执行
        self.load_all_text(path)

    def load_all_text(self,path):
        pathlist = [os.path.join(path, x) for x in os.listdir(path)]
        # pathlist=os.listdir(path)
        print(pathlist)
        # 首先进行遍历每个文件拿到统计数据
        for text_path in pathlist:
            # textpath=os.path.join(path,textpath)
            self.load_one_text(text_path)
        # 然后进行合并，合并完的编码会更新到每个单独的文件中的码值里面
        self.merge_all_state()
        # 然后每个文件都进行解码之后做对比
        for text_path in pathlist:
            self.decode(text_path)
        # 最后进行硬解码之后做对比
        self.encode()

    def load_one_text(self,path):
        with open(path,'r',encoding='utf-8') as f:
            # 这里的作用是将每个传入的文件中的字进行解码然后两两计算出现的次数并且存储起来
            sentence=f.read().strip()
            self.origin_sentence[path]['origin']=sentence
            encoding_result = sentence.encode('utf-8')
            encoding_result = list(map(int, encoding_result))
            self.origin_elem_of_dict[path] = encoding_result
            self.new_elem_of_dict[path] = encoding_result
        f.close()

    def get_stats(self):
        '''
        将所有的新元素都进行一次合并重新统计
        :return:
        '''
        result=defaultdict(dict)
        for path,elem in self.new_elem_of_dict.items():
            for tuple in zip(elem, elem[1:]):
                result[tuple]=result.get(tuple,0)+1
        return result

    def merge_state(self,tuple,target):
        # 遍历每个文件中的编码信息 这里更新的是new这个字典中的码值 ,用于和之前的做对比
        for index_path,one_dict in self.new_elem_of_dict.items():
            new_dict=[]
            # 这里是从第一个索引开始找全部相同能匹配的上的二元组，然后进行替换
            index=0
            # 只要没循环到最后一个元组，就一直循环查找
            while index<len(one_dict):
                # z这里的判断逻辑是首先得保证有两个元素=可以被锁定，如果查询到最后一个元素则没必要和二元组比较，直接放入
                # 另外就是比较当前元素和元组中第一个元素匹配，当前元素之后和元组中的第二个元素匹配
                if index<len(one_dict)-1 and one_dict[index]==tuple[0] and one_dict[index+1]==tuple[1]:
                    new_dict.append(target)
                    # 这里是因为将两个元素合并为了一个，所以一次性跳过两个
                    index+=2
                else:
                    # 否则就原值放回去
                    new_dict.append(one_dict[index])
                    index+=1
            # 更新被编码之后的
            self.new_elem_of_dict[index_path]=new_dict

    def merge_all_state(self):
        size=self.max_vocab_size-256
        for i in range(size):
            # 首先进行一次统计当前最多的元组数量
            self.all_dict=self.get_stats()
            # 拿出次数最多的组合进行合并
            target_tuple=max(self.all_dict,key=self.all_dict.get)
            # 合并后的编码
            target=256+i
            print(f" 取出的元组为 {target_tuple} ，当前数量为{self.all_dict[target_tuple]} 合并后的编码为{target} ")
            # 针对每一个文件中的值进行合并
            self.merge_state(target_tuple,target)
            # 记录被编码之后的值
            self.merge_drct[target_tuple]=target

    def decode(self,path):
        # 这里是解码的过程
        # 首先拿到常规每个编码的字节流
        vocab={ index:bytes([index]) for index in range(256)}
        # 然后将合并后的编码反编译拼接回原来的字节流
        for (p0, p1), idx in self.merge_drct.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # 最后将字节流解码为字符串
        tokens = b"".join(vocab[index] for index in self.new_elem_of_dict[path])
        # 反编译为原来的字符串
        text = tokens.decode("utf-8", errors="replace")
        self.origin_sentence[path]['decode'] = text
        flag=self.origin_sentence[path]['decode']==self.origin_sentence[path]['origin']
        print(f'{path} 文件 解码之后的当前文本和原文本比对结果为 {flag} ')

    def encode_get_state(self,encoding_result):
        '''
        专门为encode硬编码实现的获取二元组统计的算法
        todo 此算法与上面的算法有功能重叠的部分，可进行优化
        :param path:
        :return:
        '''
        result = defaultdict(dict)
        for tuple in zip(encoding_result, encoding_result[1:]):
            result[tuple]=result.get(tuple,0)+1
        return result

    def encode_merge_state(self,one_dict,tuple,target):
        '''
        这里也是专门为硬解析写的方法
        todo 和上面的方法功能有重叠的部分，可优化为同样的方法
        :param one_dict:
        :param tuple:
        :param target:
        :return:
        '''
        new_dict = []
        # 这里是从第一个索引开始找全部相同能匹配的上的二元组，然后进行替换
        index = 0
        # 只要没循环到最后一个元组，就一直循环查找
        while index < len(one_dict):
            # z这里的判断逻辑是首先得保证有两个元素=可以被锁定，如果查询到最后一个元素则没必要和二元组比较，直接放入
            # 另外就是比较当前元素和元组中第一个元素匹配，当前元素之后和元组中的第二个元素匹配
            if index < len(one_dict) - 1 and one_dict[index] == tuple[0] and one_dict[index + 1] == tuple[1]:
                new_dict.append(target)
                # 这里是因为将两个元素合并为了一个，所以一次性跳过两个
                index += 2
            else:
                # 否则就原值放回去
                new_dict.append(one_dict[index])
                index += 1
        # 更新被编码之后的
        return new_dict

    def encode(self):
        '''
        这段代码的作用是利用已经整合好的合并的原编码和新的编码合集，来对文件中的内容直接进行硬编码
        这个方法运行的前提是已经进行过一次bpe算法或者是将被合并的编码和新的编码收集起来才能进行编码
        :return:
        '''
        self.encode_elem_of_dict=defaultdict(dict)
        for path,text in self.origin_sentence.items():
            # 拿到原始的文本然后进行解码
            sentence=text['origin']
            encoding_result = sentence.encode('utf-8')
            encoding_result = list(map(int, encoding_result))
            # 然后根据合并后的编码进行硬编码
            while len(encoding_result)>= 2 :
                # 首先每次都进行统计一次当前的值
                all_dict=self.encode_get_state(encoding_result)
                # 然后根据统计结果进行合并
                tuple=min(all_dict,key=lambda x: self.merge_drct.get(x, float("inf")))
                '''
                代码中是拿着最后已经合并好的元组和对应的新码值去编码，
                所以进行编码合并的时候优先选择最开始被命中合并编码的,
                因为后面码值越大不排除这个码值可能是合并之后再次合并的可能性
                所以在使用已经总结好的元组对去编码的时候就从一开始被替换的元组对开始编译
                '''
                # 这个判断条件说明该合并的已经合并完了，没有可以继续合并的元组对，则直接跳出编译
                if tuple not in self.merge_drct:
                    break
                # 拿到元组合并之后的新编码
                index=self.merge_drct[tuple]
                # 进行替换
                encoding_result=self.encode_merge_state(encoding_result,tuple,index)
            # 然后将硬编码完成的码值保存起来做对比
            self.encode_elem_of_dict[path]=encoding_result
        # 整个循环完成之后说明硬编码完成了，进行对比
        for path,text in self.new_elem_of_dict.items():
            # 进行对比
            flag=self.encode_elem_of_dict[path]==text
            print(f'{path} 文件 硬 解码之后的当前文本和原文本比对结果为 {flag} ')


# test=MyDiyBpe('Heroes/上古巨神.txt',270)
# test.load_one_text('Heroes/上古巨神.txt')
# print("-------------")
# test.merge_all_state()
# test.decode('Heroes/上古巨神.txt')

test=MyDiyBpe('Heroes',300)
