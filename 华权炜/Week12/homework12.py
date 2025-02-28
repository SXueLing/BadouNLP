	           位置编码	transformer结构	多头机制	  ff层设计	归一化层选择	      激活函数	是否使用bias
baichuan2-7b	RoPE	      串行	      传统方式	  gated形式	RMSnorm/pre norm	   SiLU	      无bias
baichuan2-13b	Alibi	      串行	      传统方式	  gated形式	RMSnorm/pre norm	   SiLU	      无bias
chatglm2	    RoPE	      串行	     multi query	gated形式	RMSnorm/pre norm	   SiLU	      qkv有bias，其他线性层无bias
llama2	      RoPE	      串行	     multi query	gated形式	RMSnorm/pre norm	   SiLU	      无bias
moss	        RoPE	      平行	     传统方式	    传统方式	LayerNorm	          gelu_new	  sa无bias, ff有bias
qwen	        RoPE	      串行	     传统方式	    gated形式	RMSnorm/pre norm	   SiLU	      qkv有bias，其他线性层无bias
chatglm3	    RoPE	      串行	     multi query	gated形式	RMSnorm/pre norm	   SiLU	      qkv有bias，其他线性层无bias
DBRX	        RoPE	      并行	     传统方式	    gated形式	RMSnorm/pre norm	   gelu_new	  无bias
deepseek	    RoPE	      并行	        MLA	      传统方式	RMSnorm/post norm	   SiLU	      无bias
gemma	        RoPE	      并行	     multi query	gated形式	RMSnorm/pre norm/post norm	GeGLU	无bias
grok1	        RoPE	      并行	     传统方式	    传统方式	LayerNorm/post norm	gelu_new	   无bias
Mixtral	      RoPE	      并行	     grouped query	MoE	    RMSnorm/pre norm	   SiLU	        无bias
