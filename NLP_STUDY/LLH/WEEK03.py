import torch
impor torch.nn as nn

# 定义字符集大小
num_chars = 128

# 模型
class RNNModel(nn.Module):
  def_init_(self, input_size, hidden_size, output_size):
  super(RNNModule, self)._init_()
  self.rnn = nn.RNN(input_size, hidden_size, batch_first = True)
  self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = sel.rnn(x)
    out = self.fc(out[:,-1,:])
    return out

specific_char = 'a'


# 数据预处理函数
def preprocess_data(strings):
    input_sequences = []
    targets = []
    max_length = 0
    for string in strings:
        char_indices = []
        pos = string.find(specific_char)
        if pos == -1:
            pos = len(string)
        for char in string:
            char_indices.append(ord(char))
        input_sequences.append(char_indices)
        targets.append(pos)
        max_length = max(max_length, len(string))

    for sequence in input_sequences:
        while len(sequence) < max_length:
            sequence.append(0)

    input_sequences = torch.tensor(input_sequences, dtype=torch.float32).unsqueeze(2)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_sequences, targets


# 示例字符串数据
strings = ["abcde", "fghija", "klmnop"]
input_sequences, targets = preprocess_data(strings)

# 定义模型参数
input_size = 1
hidden_size = 16
output_size = max([len(string) for string in strings]) + 1

model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_sequences)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


