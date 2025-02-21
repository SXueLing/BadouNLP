import torch
from loader import load_data

"""
模型的效果分析

[description]
"""

class Evaluate(object):
    """docstring for Evaluate"""
    def __init__(self, config,model,logger):
        super(Evaluate, self).__init__()
        self.config = config
        self.logger = logger
        self.model = model
        self.valid_data = load_data(config["volid_data_path"],config,shuffle=False)

        self.train_data = load_data(config["train_data_path"],config)
        self.stats_dict = {"correct":0,"wrong":0}  # 收集测试结果

    def knowledgebase_to_vector(self):
        self.question_index_to_standard_index = {} # {问题：标准问}
        self.question_ids = [] # [问题索引]
        # 遍历训练集中知识库中的{标准问：[变形问]}
        # 知识库问题与标准进行一个映射联系
        for standard_question_index,question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 
                self.question_index_to_standard_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)

        with torch.no_grad():
            question_matrix = torch.stack(self.question_ids,dim=0)
            if torch.cuda.is_available():
                question_matrix = question_matrix.cuda()
            self.knwb_vectors  = self.model(question_matrix)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors,dim=-1)

        return

    def eval(self,epoch):
        # 主调用函数
        # 那所有测试集的数据进行验证，每一轮的模型，需要提前设定
        self.logger.info(f"开始测试第{epoch}轮的模型效果")
        self.stats_dict = {"correct":0,"wrong":0}
        self.model.eval()
        self.knowledgebase_to_vector()
        for index,batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [data.cuda() for data in batch_data]
            input_ids,labels = batch_data  # 这里的label是[index],还是index,应该是张量
            with torch.no_grad():
                test_question_vectors = self.model(input_ids)
            self.write_stats(test_question_vectors,labels)
        self.show_stats()


    def write_stats(self,test_vectors,labels):
        assert len(test_vectors)==len(labels)
        for test_vectors,label in zip(test_vectors,labels):
            # print(self.knwb_vectors.shape)
            result = torch.mm(torch.unsqueeze(test_vectors,dim=0),self.knwb_vectors.t())
            hit_index = int(torch.argmax(result.squeeze()))
            hit_index = self.question_index_to_standard_index[hit_index]
            if int(hit_index) == label.item():
                self.stats_dict["correct" ] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return



