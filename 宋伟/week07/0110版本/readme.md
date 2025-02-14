**不同模型文本分类效果对比**

# 文本分类项目

## 项目简介
本项目使用 PyTorch 实现文本分类，比较了三种模型：LSTM、CNN 和 RNN。

## 项目结构

```bash
text_classification_project/ 

├── config.py # 配置文件 

├── evaluate.py # 模型评估代码 

├── loader.py # 数据加载与预处理 

├── model.py # 模型定义 

├── main.py # 主程序 

├── README.md # 项目说明文档 

└── requirements.txt # 依赖库
```



## 环境依赖

安装以下依赖：
```bash
pip install -r requirements.txt
数据准备
在  主目录下准备 train.csv 和 val.csv 文件，这里暂时使用`文本分析训练.csv` 文件需包含 review 和 label 两列。

运行项目
运行以下命令：

python main.py
模型说明
LSTM：长短时记忆网络
CNN：卷积神经网络
RNN：循环神经网络
输出结果
每个模型的分类报告和训练过程将在控制台打印。

---

### **运行步骤**
1. 确保数据已准备好并存储在 `./` 目录下。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
运行主程序：
python main.py
```



## 结果分析

> 这里展示统计基础对比的数据

| model     | precision | Recall | F1   | support | Accuracy |
| --------- | --------- | ------ | ---- | ------- | -------- |
| LSTM(avg) | 0.75      | 0.69   | 0.70 | 11987   | 0.76     |
| CNN(avg)  | 0.82      | 0.79   | 0.80 | 11987   | 0.83     |
| RNN(avg)  | 0.33      | 0.50   | 0.40 | 11987   | 0.67     |



