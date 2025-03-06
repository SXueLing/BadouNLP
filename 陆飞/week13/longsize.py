def find_longest_sentence_length(file_path):
    max_length = 0
    current_length = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                current_length = 0

    # 检查最后一个句子
    if current_length > max_length:
        max_length = current_length

    return max_length

# 文件路径
file_path = "F:\\AI课\\week9 序列标注问题\\ner_with_bert\\ner_data\\train"

# 计算最长句子长度
longest_sentence_length = find_longest_sentence_length(file_path)
print(f"最长句子长度: {longest_sentence_length}")
