from loader import load_test_data
import torch
from config import Config
import logging
from model import TorchModel
from loader import load_voacb
import logging
import time

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator():
    def __init__(self, config, model, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.valid_data = load_test_data(config['train_data_path'], config)
        self.stat_dict = {'正确': 0, '错误': 0}
        
    def eval(self, epoch):
        self.logger.info(f'开始第{epoch}轮测试')
        self.model.eval()
        self.stat_dict = {'正确': 0, '错误': 0}
        for index, batch_data in enumerate(self.valid_data):
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_ids)
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc
    
    def write_stats(self, labels, pred_results):
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stat_dict["正确"] += 1
            else:
                self.stat_dict["错误"] += 1
        return
    
    def show_stats(self):
        correct = self.stat_dict["正确"]
        wrong = self.stat_dict["错误"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
   
if __name__ == '__main__':
    time_start = time.time()
    Config['class_num'] = 2
    Config['vocab_size'] = len(load_voacb(Config['vocab_path']))
    torch_model = TorchModel(Config)
    # torch_model.load_state_dict(torch.load(Config['model_path'] + '/gru_model.pth'))
    # torch_model.load_state_dict(torch.load(Config['model_path'] + '/lstm_model.pth'))
    torch_model.load_state_dict(torch.load(Config['model_path'] + '/rnn_model.pth'))
    evaluator = Evaluator(Config, torch_model, logger)
    evaluator.eval(1)
    time_end = time.time()
    logger.info("总耗时：%f" % (time_end - time_start))