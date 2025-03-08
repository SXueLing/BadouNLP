import torch
from loader import loader


# 验证模型训练效果类
class Evaluation:
    def __init__(self, config, model):
        self.model = model
        self.config = config
        self.data = loader(self.config["eval_data_path"], self.config)
        self.correct = 0
        self.wrong = 0

    def evaluator(self):
        self.correct = 0
        self.wrong = 0
        self.model.eval()
        for index, eval_data in enumerate(self.data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in eval_data]
            else:
                batch_data = eval_data
            x, y_true = batch_data
            with torch.no_grad():
                y_pred = self.model(x)
            self.get_result(y_pred, y_true)
        acc = self.result_show()
        return acc

    def get_result(self, y_pred, y_true):
        for y_p, y_t in zip(y_pred, y_true):
            if torch.argmax(y_p) == y_t[0]:
                self.correct += 1
            else:
                self.wrong += 1
        return

    def result_show(self):
        print("===============================")
        print(
            f"一共有{self.correct + self.wrong}个样本, 预测正确{self.correct}个，预测错误{self.wrong}个，正确率为{self.correct / (self.correct + self.wrong)}")
        return self.correct / (self.correct + self.wrong)
