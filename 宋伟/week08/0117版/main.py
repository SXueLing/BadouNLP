import torch
from torch import nn
from torch.optim import Adam
from config import CONFIG
from loader import get_dataloader
from model import TextEmbeddingModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# 初始化模型和数据
model = TextEmbeddingModel(CONFIG['model_name'], CONFIG['embedding_dim'])
data_loader = get_dataloader(CONFIG['data_path'], CONFIG['batch_size'])
triplet_loss = nn.TripletMarginLoss(margin=CONFIG['margin'], p=2)
optimizer = Adam(model.parameters(), lr=CONFIG['learning_rate'])

# 训练循环
for epoch in range(CONFIG['num_epochs']):
    model.train()
    total_loss = 0
    for anchor, positive, negative in data_loader:
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)

        loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Loss: {total_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), CONFIG['save_model_path'])