###############训练数据拆分 start##############################
import pandas as pd
from sklearn.model_selection import train_test_split
# 读取 CSV 文件到 DataFrame
df = pd.read_csv('文本分类练习.csv')

# 始化分词器 确定模型的max_length
tokenizer = BertTokenizer.from_pretrained(r"bert-base-chinese")
# 假设 df 是包含文本数据的 DataFrame
text_lengths = df['review'].apply(lambda x: len(tokenizer.tokenize(x)))

#确定模型的max_length
print(f"Min length: {text_lengths.min()}")
print(f"Max length: {text_lengths.max()}")
print(f"Mean length: {text_lengths.mean()}")
print(f"Median length: {text_lengths.median()}")
print(f"95th percentile length: {text_lengths.quantile(0.95)}")

# 分别筛选出 label == 0 和 label == 1 的数据
df_label_0 = df[df['label'] == 0]
df_label_1 = df[df['label'] == 1]
# 分割 label == 0 的数据集为训练集和测试集 (90% 训练, 10% 测试)
train_0, test_0 = train_test_split(df_label_0, test_size=0.1, random_state=42)
# 分割 label == 1 的数据集为训练集和测试集 (90% 训练, 10% 测试)
train_1, test_1 = train_test_split(df_label_1, test_size=0.1, random_state=42)

# 合并训练集和测试集
train_data = pd.concat([train_0, train_1])
test_data = pd.concat([test_0, test_1])

# 打乱合并后的训练集和测试集以确保随机性
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 将筛选后的数据写入 JSON 文件，默认格式是记录列表
train_data.to_json('train_data.json', orient='records', lines=True, indent=None)
test_data.to_json('test_data.json', orient='records', lines=True, indent=None)
###############训练数据拆分 完成###########################################

#########数据加载loader.py修改部分代码如下############################

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: 0, 1: 1}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                #print(line)
                line = json.loads(line)
                tag = line["label"]
                label = self.label_to_index[tag]
                title = line["review"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

############################训练结果对比####################################################################

模型       | 学习率  | hidden_size |  batch_size |  pooling_style | num_layers | max_length | epoch |  acc  | 预测1199条耗时
bert         0.001        256            64              avg            12           70         15    0.333      12分钟
bert         0.001        256            64              avg            12           70         10    0.333      12分钟
bert         0.001        256            64              avg            12           70         3     0.879      12分钟
gated_cnn    0.001        128            64              max             2           70         15    0.871      2秒
gated_cnn    0.001        128            64              avg             2           70         15    0.874      2秒
gated_cnn    0.0001       128            64              max             2           70         15    0.893      2秒
gated_cnn    0.0001       128            64              avg             2           70         15    0.869      2秒
lstm         0.001        128            64              max             2           70         15    0.874      8秒
lstm         0.0001       128            64              max             2           70         15    0.875      8秒
