|模型   | 位置编码 |Transformer结构      | 多头机制 |FF层设计      | 归一化层选择 |激活函数      | 是否使用Bias |
| :----:|   :----:   |    :----:   |    :----:   |    :----:   |    :----:   |    :----:   |
|Baichuan2-7B/13B| RoPE (旋转编码)|Pre-LayerNorm|标准多头注意力|Gated Linear Unit (GLU)|RMSNorm|	SwiGLU|线性层无Bias|
|ChatGLM2|RoPE (改进版)|Post-LayerNorm|标准多头注意力|GLU结构（GeGLU）|LayerNorm|	GeGLU|部分层使用Bias|
|LLaMA2|RoPE|Pre-LayerNorm|标准多头注意力|SwiGLU（中间维度扩展8x）|RMSNorm|	SwiGLU|线性层无Bias|
|MOSS|ALiBi (相对位置)|Post-LayerNorm|多查询注意力 (MQA)|标准FFN（ReLU扩展）|LayerNorm|	ReLU|部分层使用Bias|
|Qwen-7B| RoPE|Pre-LayerNorm|标准多头注意力|Gated FFN（中间维度扩展8x）|RMSNorm|	GELU|线性层无Bias|
|DeepSeek-v3| RoPE|Pre-LayerNorm|标准多头注意力|Gated FFN（SwiGLU）|RMSNorm|	SwiGLU|线性层无Bias|
|Mixtral| RoPE|Pre-LayerNorm|标准多头注意力|SwiGLU（MoE结构扩展）|RMSNorm|	SwiGLU|线性层无Bias|git