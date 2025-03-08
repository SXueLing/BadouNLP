import torch
import math
import numpy as np
from transformers import BertModel

bert = BertModel.from_pretrained("./pretrain_model/bert-base-chinese")

print(sum([p.numel() for p in bert.parameters() if p.requires_grad]))

