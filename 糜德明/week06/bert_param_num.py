vocab = 21128               # 词表数目
embedding_size = 768        # embedding维度
num_layers = 1              # 隐藏层层数
# 'embeddings.word_embeddings.weight'   					vocab * embedding_size【词表大小】
# 'embeddings.position_embeddings.weight'					512 * embedding_size【语序】
# 'embeddings.token_type_embeddings.weight' 				2 * embedding_size【语句来源】
# 'embeddings.LayerNorm.weight'  							embedding_size【归一化】
# 'embeddings.LayerNorm.bias' 							embedding_size【归一化】
# 'encoder.layer.0.attention.self.query.weight' 			embedding_size * embedding_size【Self-Attention Q】
# 'encoder.layer.0.attention.self.query.bias' 			embedding_size【Self-Attention Q】
# 'encoder.layer.0.attention.self.key.weight' 			embedding_size * embedding_size【Self-Attention K】
# 'encoder.layer.0.attention.self.key.bias' 				embedding_size【Self-Attention K】
# 'encoder.layer.0.attention.self.value.weight'			embedding_size * embedding_size 【Self-Attention V】
# 'encoder.layer.0.attention.self.value.bias' 			embedding_size【Self-Attention V】
# 'encoder.layer.0.attention.output.dense.weight' 		embedding_size * embedding_size【线性层】
# 'encoder.layer.0.attention.output.dense.bias' 			embedding_size【线性层】
# 'encoder.layer.0.attention.output.LayerNorm.weight' 	embedding_size【计算残差】
# 'encoder.layer.0.attention.output.LayerNorm.bias' 		embedding_size【计算残差】
# 'encoder.layer.0.intermediate.dense.weight' 			embedding_size * (embedding_size * 4)  【Feed Forward】
# 'encoder.layer.0.intermediate.dense.bias' 				embedding_size * 4【Feed Forward】
# 'encoder.layer.0.output.dense.weight' 					(embedding_size * 4) * embedding_size【Feed Forward】
# 'encoder.layer.0.output.dense.bias' 					embedding_size【Feed Forward】
# 'encoder.layer.0.output.LayerNorm.weight' 				embedding_size【计算残差】
# 'encoder.layer.0.output.LayerNorm.bias' 				embedding_size【计算残差】
# 'pooler.dense.weight' 									embedding_size * embedding_size【CLS】
# 'pooler.dense.bias'										embedding_size【CLS】
all_params = num_layers * embedding_size * (vocab + 512 + 2 + 1 + 1 + embedding_size + 1 + embedding_size + 1 + embedding_size + 1 + embedding_size + 1 + 1 + 1 + embedding_size * 4 + 4 + embedding_size * 4 + 1 + 1 + 1 + embedding_size + 1)

print("diy计算参数个数为%d" % all_params)
