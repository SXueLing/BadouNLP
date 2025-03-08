
可训练的参数：Embedding + self-Attention + Feed Forward
一层的参数量：
Token Embeddings:  Voc * 768
Segment Embeddings: (1 || 2) * 768  (可有可无)
Position Embeddings: 512 * 768   (最大)
Q:  768*768 + L*768  (wx + b)
K:  768*768 + L*768  (wx + b)
V:  768*768 + L*768  (wx + b)
Liner: 768*768 + L*768
Layer_norm: 768*768 + L*768
Feed Forward:  (768*3072 + L*3072) + (3072*768 + L*768)   (两个wx+b)
Layer_norm: 768*768 + L*768
