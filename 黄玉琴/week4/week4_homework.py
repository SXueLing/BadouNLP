import jieba

Dict = {"经常":0.2,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}
def calc_dag(sentence):
        HYQ={}
        N=len(sentence)
        for k in range(N):
            list=[]
            i=k
            frag=sentence[k]
            while i<N:
                if frag in Dict:
                    list.append(i)
                i=i+1
                frag=sentence[k:i+1]
            if not list:
                list.append(i)
            HYQ[k]=list
        return HYQ
sentence='经常有意见分歧'
print(calc_dag(sentence))

class DAGDecode:
    def __init__(self,sentence):
        self.sentence=sentence
        self.HYQ=calc_dag(sentence)
        self.length=len(sentence)
        self.unfinish_path=[[]]
        self.finish_path=[]

    def decode_next(self,path):
        path_length=len(''.join(path))
        if path_length==self.length:
            self.finish_path.append(path)
            return
        candiates=self.HYQ[path_length]
        new_patchs=[]
        for candiate in candiates:
            new_patchs.append(path+[self.sentence[path_length:candiate+1]])
        self.unfinish_path+=new_patchs
        return
    def decode(self):
        while self.unfinish_path !=[]:
            path=self.unfinish_path.pop()
            self.decode_next(path)

sentence = "经常有意见分歧"
dd = DAGDecode(sentence)
dd.decode()
print(dd.finish_path)

