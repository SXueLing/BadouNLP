#文本长度L  维数V
#Bert结构
#Embedding层:
#Token Embedding 参数量: L*V
#Segment Embedding 参数量: L*V
#Position Embedding 参数量: L*V
#加和Layer Normalization : L*V

#Transformer结构:
#Self-attention:
#Q、K转置、V: 3*L*V
#两层LayerNorm: 2*L*V
#Free Forward:
#线性层1+激活层+线性层2: 3*L*V

#Bert结构参数量: 4*L*V+12*(3*L*V+2*L*V+3*L*V)
