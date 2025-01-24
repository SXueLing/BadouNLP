正样本数	4000					
负样本数	7987					
长度中位数	17					
最长文本长度	463					
平均长度	25.05					
model	    learning_rate	hidden_size	epoch	batchsize	  acc	    time(per 100 data)
双向lstm	0.01	            64	      10	     32	    0.6715	  0.05s
rnn	      0.001	            128	      100	     32	    0.5342	  0.004s
cnn	      0.001	            100	      10	     64	    0.5475	  0.005s
