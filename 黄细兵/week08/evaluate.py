import torch

from sentence_encoder.loder import load_data


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config['valid_data_path'], config, shuffle=True)
        self.train_data = load_data(config['train_data_path'], config, shuffle=True)
        self.stats_dict = {"correct":0, "wrong":0}

    # 所有问题库问题转向量
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        items = self.train_data.dataset.knwb.items()
        for standard_question_index, question_ids in items:
            for question_id in question_ids:
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            #将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct":0, "wrong":0}
        self.knwb_to_vector()
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, index = batch_data
            with torch.no_grad():
                test_question_vectors = self.model(input_id)
            self.write_stats(test_question_vectors, index)
        self.show_stats()
        return

    def write_stats(self, test_question_vectors, indexs):
        assert len(indexs) == len(test_question_vectors)
        for test_question_vector, index in zip(test_question_vectors, indexs):
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res).squeeze())
            hit_index = self.question_index_to_standard_question_index[hit_index]
            if int(hit_index) == int(index):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return
