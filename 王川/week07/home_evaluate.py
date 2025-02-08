import torch
import time
from test_loader import load_data


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info("开始测试%d轮模型" %(epoch))
        self.model.eval()
        self.state_dict = {"correct": 0, "wrong": 0}
        start_time = time.time()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input, target = batch_data
            with torch.no_grad():
                predict_res = self.model(input)
                self.write_state(predict_res, target)
        end_time = time.time()
        use_time = 10000 * (end_time - start_time)/(self.state_dict["correct"] + self.state_dict["wrong"])
        acc = self.show_state(use_time)

    def write_state(self, predict_res, target):
        for y_p, y_t in zip(predict_res, target):
            pre_label = y_p.argmax()
            if int(pre_label) == int(y_t):
                self.state_dict["correct"] += 1
            else:
                self.state_dict["wrong"] += 1

    def show_state(self, use_time):
        correct = self.state_dict["correct"]
        wrong = self.state_dict["wrong"]
        self.logger.info("预测总数量为：%d" %(correct + wrong))
        self.logger.info("预测正确的数量为：%d，错误的数量为：%d" %(correct, wrong))
        self.logger.info("预测正确率为：%f" %(correct/(correct + wrong)))
        self.logger.info("预测100条数据所需时间为：%fms" %(use_time))
        self.logger.info("----------------")
        return correct / correct + wrong
