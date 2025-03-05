# 使用BERT预训练模型实现SFT

具体包括
1. 使用`bert+mask`s实现自回归的训练
2. 训练一个`sft`任务

**代码细节**

1. 输入数据：`<cls>+prompt+<sep>+answer+<sep>`,
2. 标签数据：`[-1]*len(prompt)+[-1]+answer+[-1]`