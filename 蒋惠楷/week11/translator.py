import torch
import torch.nn.functional as F

class Translator:
    def __init__(self, model, beam_size, output_max_length, pad_idx, start_idx, end_idx):
        self.model = model
        self.beam_size = beam_size
        self.output_max_length = output_max_length
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx

    def translate_sentence(self, input_ids):
        return self.greedy_decode(input_ids)

    def greedy_decode(self, input_ids):
        """
        贪心解码
        """
        device = input_ids.device

        # 确保input_ids是二维的 [batch_size, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # 为输入加上批次维度

        # 初始化输入的嵌入层
        decoder_input = torch.full((1, 1), self.start_idx, dtype=torch.long).to(device)  # 起始token

        # 用来存储生成的token
        generated_tokens = []

        # 逐步生成token直到达到最大长度或遇到结束标记
        for _ in range(self.output_max_length):
            # 将生成的token传入模型进行预测
            decoder_output, _ = self.model(input_ids, attention_mask=None, decoder_input_ids=decoder_input, lm_mask=None)
            logits = decoder_output[:, -1, :]  # 取最后一个位置的logits
            probs = F.softmax(logits, dim=-1)  # 使用softmax计算概率分布

            # 使用贪心策略，选择概率最大的token
            next_token = torch.argmax(probs, dim=-1)

            # 如果选择的token是结束符，停止解码
            if next_token == self.end_idx:
                break

            # 向decoder输入添加新生成的token
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

            # 将生成的token添加到生成的tokens列表中
            generated_tokens.append(next_token.item())

        return generated_tokens


    def beam_search_decode(self, input_ids):
        """
        Beam Search
        """
        device = input_ids.device

        # 确保input_ids是二维的 [batch_size, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # 为输入加上批次维度

        decoder_input = torch.full((1, 1), self.start_idx, dtype=torch.long).to(device)  # 起始token

        # 初始化Beam Search的状态
        sequences = [[decoder_input, 0]]  # 序列和对应的累积得分
        all_sequences = []  # 存储所有beam的生成结果

        # 逐步生成token直到达到最大长度
        for _ in range(self.output_max_length):
            all_candidates = []
            for seq, score in sequences:
                # 将当前序列传入模型进行预测
                decoder_output, _ = self.model(input_ids, attention_mask=None, decoder_input_ids=seq, lm_mask=None)
                logits = decoder_output[:, -1, :]  # 取最后一个位置的logits
                probs = F.softmax(logits, dim=-1)  # 使用softmax计算概率分布

                # 选择top_k个候选token
                topk_scores, topk_tokens = torch.topk(probs, self.beam_size, dim=-1)

                # 对每个候选token，更新生成的序列
                for i in range(self.beam_size):
                    candidate_seq = torch.cat([seq, topk_tokens[:, i].unsqueeze(0)], dim=1)
                    candidate_score = score - torch.log(topk_scores[:, i])
                    all_candidates.append([candidate_seq, candidate_score])

            # 按得分排序，保留最好的beam_size个序列
            sequences = sorted(all_candidates, key=lambda x: x[1])[:self.beam_size]

            # 检查是否所有的序列都以end_idx结尾
            # 修改了此处的索引方式
            if all(seq[0][-1] == self.end_idx for seq, _ in sequences):
                break

        # 选择得分最高的序列作为最终输出
        best_sequence = sequences[0][0]
        return best_sequence.squeeze(0).cpu().numpy()
