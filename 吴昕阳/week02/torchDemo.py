import torch 
import torch.nn as nn
import numpy as np


class TorchDemo(nn.Module):
    def __init__(self,input_size):
        super(TorchDemo, self).__init__()
        self.layer = nn.Linear(input_size, 5)
        #self.activation = nn.Softmax
        self.loss = nn.CrossEntropyLoss #因为交叉熵函数包含了softmax函数
        #self.loss = nn.functional.cross_entropy
    def forward(self, x,y=None):
        y_pred = self.layer(x)
        if y is not None:
            return self.loss(y_pred, y)
          
        else:
            return y_pred   

def build_simple():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index  

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_simple()
        X.append(x)
        Y.append(y) #这个为啥不用加【y】=>因为 y 是一个整数，
                    #直接追加到列表中即可；而加 [y] 会导致 Y 
                    # 的结构变成嵌套列表（[[0], [1], ...]），不符合分类标签的要求
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evalute(model):
    model.eval()
    total_sample_num = 100
    x,y = build_dataset(total_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_pred,y_t in zip(y_pred,y):
            #if torch.argmax(y_pred) == y_t:
            if y_pred.argmax() == y_t:
                correct += 1
            else:
                wrong += 1
    return correct / (correct + wrong)


def main():
   epoch_num = 10
   total_sample_num = 1000
   batch_size = 20
   input_size = 5
   learning_rate = 0.001
   log = []
   model = TorchDemo(input_size)
   optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
   train_x,train_y = build_dataset(total_sample_num)
   for epoch in range(epoch_num):
       model.train()
       watch_loss = []
       for batch_idx in range(total_sample_num // batch_size):
           x = train_x[batch_idx * batch_size : (batch_idx + 1) * batch_size]
           y = train_y[batch_idx * batch_size : (batch_idx + 1) * batch_size]
           optim.zero_grad()
           loss = model.forward(x,y)
           loss.backward()
           optim.step()
           watch_loss.append(loss.item())
       acc = evalute(model)    
       log.append([acc, float(np.mean(watch_loss))])  

   torch.save(model.state_dict(), 'model.pt')
   return
      
def predict(model_path,input_vec):
    input_size = 5
    model = TorchDemo(input_size)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    with torch.no_grad():
        output = model.forward(torch.FloatTensor(input_vec))

