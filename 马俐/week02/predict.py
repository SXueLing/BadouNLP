from train_model import Model
import torch
from build_dataset import build_dataset

# 使用训练好的模型做预测
def predict(model_path, input_vec):

    model = Model()
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward_pred(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print(f"输入：{vec}, 预测类别：{res}")  # 打印结果

if __name__ ==  "__main__":

    model_path = "model.pt"
    input, y = build_dataset(5)
    predict(model_path, input)