          muli-head共享方式（multi_head，grouped_query，multi_query）	attention结构	           归一化层位置选择	 激活函数	moe结构
llama	       multi_head，grouped_query	                              传统transformer block	     Pre   	          RMSNorm	FALSE
chatglm3	       multi_query	                                        传统transformer block	     Pre	            RMSNorm	FALSE
chatglm2	       multi_head	                                          传统transformer block	     Pre	            RMSNorm	FALSE
baichuan	       multi_head	                                          传统transformer block	     Pre	            RMSNorm	FALSE
baichuan2	       multi_head	                                          传统transformer block	     Pre	            RMSNorm	FALSE
dbrx	           grouped_query	                                      传统transformer block	     Pre	           LayerNorm	TRUE
deepseekv3	     grouped_query	                                      传统transformer block	     Pre	            RMSNorm	TRUE
gemma	           grouped_query	                                      传统transformer block	     Pre	           GemmaRMSNorm	TRUE
grok1	           grouped_query	                                      传统transformer block	     Sandwish	         RMSNorm	TRUE
mixtral	         grouped_query	                                      传统transformer block	     Pre	           MixtralRMSNorm	TRUE
moss	           multi_head	                                                GPTJ	               Pre	            LayerNorm	FALSE
qwen	           multi_head	                                          传统transformer block	     Pre	             RMSNorm	FALSE
