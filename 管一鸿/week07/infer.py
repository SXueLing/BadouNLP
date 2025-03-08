import time
import torch
from tqdm import tqdm  # 用于显示进度条
from torch.utils.data import DataLoader
from loader import DataGenerator,load_data
from model import TorchModel
from transformers import BertTokenizer
import os
import logging
import csv
from utils import get_model_path_from_folder
from config import Config

pred_dict = {
    0:"差评",
    1:"好评"
}

class Infer:
    def __init__(self, config, model_path, hidden_size,logger=None):
        self.config = config
        self.config["batch_size"] = 1 # 计算100次所用时间时调整为1
        self.config["hidden_size"] = hidden_size
        self.model_path = model_path
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))  # 加载已训练模型
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=True)
        self.logger = logger if logger else logging.getLogger(__name__)
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])

    def infer(self, num_iterations=100):
        """
        执行推理，返回预测结果并计算推理时间。
        :param num_iterations: 执行推理的次数，默认为100
        """
        self.logger.info("开始推理...")
        cuda_flag = torch.cuda.is_available()
        if cuda_flag:
            self.model = self.model.cuda()
        self.model.eval()  # 将模型设置为评估模式

        # 记录总时间
        start_time = time.time()

        # 存储结果
        all_predictions = []
        all_input_ids = []
        data_count = 0
        for index, batch_data in enumerate(self.valid_data):
            if data_count == num_iterations:
                break
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
                input_ids, _ = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
                with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                    pred_results = self.model(input_ids)  # 获取模型预测结果


                # 保存输入ID和预测结果
                all_input_ids.extend(input_ids.cpu().numpy())
                all_predictions.extend(pred_results)

                # 打印进度
                if index % 10 == 0:
                    self.logger.info(f"推理进度: {index}/{num_iterations}")
                data_count += 1

        # 计算完成 100 次推理所需的总时间
        end_time = time.time()
        total_time = end_time - start_time  # 计算总时间
        self.logger.info(f"完成 {num_iterations} 次推理，总耗时: {total_time:.2f}秒,模型路径：{self.model_path}","infer_res.txt")
        
        # 输出预测结果到文件
        # pred_file_path = os.path.join(self.config["model_path"], f"{self.config["model_type"]}_predictions.csv")
        # self.write_predictions(all_input_ids, all_predictions,pred_file_path)

    def infer_with_text(self,title):
        self.model.eval()

        input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
        input_id = torch.LongTensor([input_id])
        with torch.no_grad():  # 禁用梯度计算，减少内存消耗
            outputs = self.model(input_id)
            predictions = torch.argmax(outputs).cpu().numpy()
        print(f"预测结果：{pred_dict[int(predictions)]}")

    def write_predictions(self, input_ids, predictions, output_file="predictions.csv"):
        """
        保存预测结果到CSV文件。
        :param input_ids: 输入的ID，用于标记预测结果的输入
        :param predictions: 模型预测的结果
        :param output_file: 输出文件路径
        """
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["input_ids", "predictions"])  # 写入列名
            for input_id, pred in zip(input_ids, predictions):
                writer.writerow([input_id, pred])

if __name__ == '__main__':
    
    dg = DataGenerator("./data/data.csv", Config)
    # 初始化日志记录器
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    folder_path = "./output"  # 传入模型所在的文件夹路径

    # 加载文件夹中的所有 .pth 文件路径
    model_paths,model_type,hidden_sizes = get_model_path_from_folder(folder_path)
    print(model_paths)
    for model_path,model_type,hidden_size in zip(model_paths,model_type,hidden_sizes):
        Config["model_type"] = model_type

        # 创建推理对象并执行推理
        infer = Infer( Config,model_path,hidden_size,logger)
        infer.infer(num_iterations=100)
    
    """
    # 直接输入文本预测
    Config["model_type"] = 'bert'
    infer = infer = Infer( Config,logger)
    text = input("请输入评价：")
    infer.infer_with_text(text)
    """