# coding:utf8
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BertForGeneration(nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain_path, add_pooling_layer=False)
        self.lm_head = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        
        # 冻结BERT参数（可选）
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(outputs.last_hidden_state)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        return logits

class TextDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.data = tokenizer(
            corpus,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
    def __len__(self):
        return self.data.input_ids.size(0) - 1
    
    def __getitem__(self, idx):
        return (
            self.data.input_ids[idx],
            self.data.input_ids[idx+1]
        )

class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.bert.device
        
    def generate(self, prompt, max_length=30, temperature=1.0, top_k=50):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
        
        # 生成缓存
        past_key_values = None
        generated = input_ids
        
        for _ in range(max_length):
            # 构建三角掩码
            seq_len = generated.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.bert(
                    generated, 
                    attention_mask=mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = self.model.lm_head(outputs.last_hidden_state[:, -1, :])
            
            # 采样策略
            logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
                
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            past_key_values = outputs.past_key_values
            
            if next_token.item() == self.tokenizer.sep_token_id:
                break
                
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

def train(corpus_path, pretrain_path):
    # 超参数配置
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 2e-4
    MAX_LENGTH = 64
    
    # 初始化组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    
    dataset = TextDataset(corpus, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = BertForGeneration(pretrain_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            inputs, labels = [x.to(device) for x in batch]
            
            # 构建下三角注意力掩码
            seq_len = inputs.size(1)
            attention_mask = torch.tril(torch.ones(inputs.size(0), seq_len, seq_len)).to(device)
            
            optimizer.zero_grad()
            loss = model(inputs, attention_mask=attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        # 生成测试
        generator = Generator(model, tokenizer)
        test_prompts = ["昔者庄周梦为胡蝶", "人工智能的未来"]
        logging.info(f"\nEpoch {epoch+1} 生成示例:")
        for prompt in test_prompts:
            generated = generator.generate(prompt, temperature=0.9, top_k=30)
            logging.info(f"{prompt} → {generated}")

if __name__ == "__main__":
    train(
        corpus_path="corpus.txt",
        pretrain_path="bert-base-chinese"
    )
