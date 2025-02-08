Bert参数计算：
1. embedding层：
   word_embedding: vocab_size * hidden_size
   position_embedding: max_position_embeddings * hidden_size
   token_type_embedding: type_vocab_size * hidden_size
   三个embedding层加起来：具体的值来自于config.json文件中
23,835,648

2. layer norm：
   LayerNorm：gamma + beta；gamma：1 * hidden_size，beta：1 * hidden_size
1,536

3. Q/W/V：12层
   Q：hidden_size * hidden_size
   W：hidden_size * hidden_size
   V：hidden_size * hidden_size
21,233,664

4. concat后经过了一个linear：12层
   hidden_size * hidden_size
7,077,888

4. feed forward：12层每层2个
全连接层的公式，其中用到了两个参数W1和W2，Bert沿用了惯用的全连接层大小设置，即4 * dmodle，为3072，因此，W1，W2大小为768 * 3072，2个为 2 * 768 * 3072。
   intermediate_size * hidden_size
   hidden_size * intermediate_size
56,623,104

5. layer norm：12层，每层2个layer norm
   LayerNorm：gamma + beta；gamma：1 * hidden_size，beta：1 * hidden_size
36864

总和：
108,808,704
