import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
n = 2                       # Enter the maximum number of sentences
vocab = 21128               # Number of vocabularies
max_sequence_length = 512   # Maximum sentence length
embedding_size = 768        # Embedding size
hide_size = 3072            # Hidden layer size
num_layers = 12             # Number of layers

# Parameters in the embedding process, where vocab * embedding_size is the vocabulary embedding parameter, max_sequence_length * embedding_size is the position parameter, and n * embedding_size is the sentence parameter
# embedding_size + embedding_sizes is the layer_norm layer parameter
embedding_parameters = vocab * embedding_size + max_sequence_length * embedding_size + n * embedding_size + embedding_size + embedding_size

# Parameters of the self_attention process, where embedding_size * embedding_size is the weight parameter, embedding_size is bias, *3 is K Q V
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

# self_attention_out parameters where embedding_size * embedding_size + embedding_size + embedding_size are the linear layer parameters of self output, and embedding_size + embedding_size are the layer_norm layer parameters
self_attention_out_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size

# Feed Forward parameters where embedding_size * hide_size + hide_size first linear layer, embedding_size * hide_size + embedding_size second linear layer,
# embedding_size + embedding_size is the layer_norm layer
feed_forward_parameters = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + embedding_size + embedding_size

# pool_fc layer parameters
pool_fc_parameters = embedding_size * embedding_size + embedding_size

# Total number of parameters
all_paramerters = embedding_parameters + (self_attention_parameters + self_attention_out_parameters + \
    feed_forward_parameters) * num_layers + pool_fc_parameters
print("The actual number of model parameters is %d" % sum(p.numel() for p in model.parameters()))
print("The number of DIY calculation parameters is %d" % all_paramerters)