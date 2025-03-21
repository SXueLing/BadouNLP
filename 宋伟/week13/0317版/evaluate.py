import torch
from loader import load_data
from collections import defaultdict
import re
import numpy as np


# 对模型效果进行测试

class Evaluator(object):
    """docstring for Evaluator"""

    def __init__(self, config, model, logger):
        super(Evaluator, self).__init__()
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config['test_data'], config, shuffle=False) # 不能打乱，不然不能获取对应的原句

    def eval(self, epoch):
        self.logger.info(f"开始测试{epoch}轮模型效果")
        self.stats_dict = {'LOCATION': defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for idx,batch_data in enumerate(self.valid_data):
            # 获取每个批次下的原句组成的列表
            sentences = self.valid_data.dataset.sentences[idx*self.config['batch_size']:(idx+1)*self.config['batch_size']]
            if torch.cuda.is_available():
                batch_data = {key:data.cuda() for key,data in batch_data.items()}

            with torch.no_grad():
                pre_result = self.model(batch_data['input_ids'])
            self.write_stats(batch_data['labels'],pre_result,sentences) # 在句子下，将标签，预测标签进行对比，进行统计
        self.show_stats()


    def write_stats(self,labels,pred,sentences):
        assert len(labels)==len(pred)==len(sentences)

        if not self.config['use_crf']:
            pred = torch.argmax(pred,dim=-1)  # 这里还是张量吧
        for true_label,pred_label,sentence in zip(labels,pred,sentences):
            if not self.config['use_crf']:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence,true_label)  # 解码出对应的命名实体,每一句中，记录{类别：实体}
            pred_entities = self.decode(sentence,pred_label)

            for key in self.stats_dict.keys():
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]['识别出实体数'] += len(pred_entities[key])

    def show_stats(self):
        F1_scores = []
        for key in self.stats_dict.keys():
            # 精确率，查准率
            precision = self.stats_dict[key]['正确识别']/(1e-5 +self.stats_dict[key]['识别出实体数'])
            # 召回率，查全率
            recall = self.stats_dict[key]['正确识别']/(1e-5 + self.stats_dict[key]['样本实体数'])
            F1 = (2*precision*recall)/(precision+recall+1e-5)
            F1_scores.append(F1)
            # 对于每一个类别存在一个F1
            self.logger.info(f"对于{key}类实体，精确率：{precision},召回率为：{recall}，F1分数为：{F1}")

        # 对于宏观上的f1，将所有类别的f1取平均处理
        self.logger.info(f"Macro-f1：{np.mean(F1_scores)}")
        # 下面计算微观上的f1

        correct_pred = sum([self.stats_dict[key]['正确识别'] for key in self.stats_dict.keys()]) # 正确预测的总数
        total_pred = sum([self.stats_dict[key]['识别出实体数'] for key in self.stats_dict.keys()]) # 预测出实体总数
        true_enti = sum([self.stats_dict[key]['样本实体数'] for key in self.stats_dict.keys()]) # 样本中的实体总数

        micro_precision = correct_pred/(1e-5+total_pred)  # 查准率，从预测结果出发
        micro_recall = correct_pred/(1e-5+true_enti) # 查全率，从样本实体出发

        micro_f1 = (2*micro_precision*micro_recall)/(micro_recall+micro_precision)
        self.logger.info(f"Micro_f1:{micro_f1}")
        self.logger.info("---------------------------------")




    def decode(self,sentence:str,labels:list):
        # 根据某一句，和标签序列，解码出实体
        # 这里的原句和label 不同太一样，label对一cls+sentence,需要一个
        sentence = '$'+ sentence
        labels = ''.join(str(x) for x in labels[:len(sentence)+1])  # 避免填充标签
        results = defaultdict(list)
        for loaction in re.finditer("04+",labels):
            s,e = loaction.span()
            results['LOCATION'].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
