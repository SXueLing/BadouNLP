import os
from collections import defaultdict
import time

def bpe_data(folder_path):
    """从指定文件夹读取数据并进行预处理"""
    pre_bpe = []
    
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在。")
        return pre_bpe
    
    if not os.path.isdir(folder_path):
        print(f"'{folder_path}' 不是一个目录。")
        return pre_bpe
    
    # 使用os.walk遍历所有文件
    file_count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_count += 1
            full_path = os.path.join(root, file)
            print(f"处理文件: {full_path}")
            
            try:
                with open(full_path, encoding="utf-8") as f:
                    for line in f:
                        # 跳过空白行
                        if not line.strip():
                            continue
                        
                        tokens = line.encode("utf-8")
                        tokens = list(map(int, tokens))
                        
                        # 可选的调试信息，可以注释掉以提高性能
                        # print('---')
                        # print(line.strip())
                        # print("字符长度:", len(line))
                        # print('---')
                        # print(tokens)
                        # print("字节长度:", len(tokens))
                        
                        pre_bpe.extend(tokens)
            except Exception as e:
                print(f"处理文件 {full_path} 时出错: {e}")
    
    print(f"共处理了 {file_count} 个文件")
    print(f"总字节数: {len(pre_bpe)}")
    return pre_bpe

def get_stats(ids):
    """统计相邻对出现的频率"""
    counts = defaultdict(int)
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    """合并指定的对"""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def bpe_func(vocab_size, ids):
    """BPE算法实现"""
    start_time = time.time()
    # 超参数：预期的最终词表大小
    num_merges = vocab_size - 256
    
    if not ids:
        print("警告: 输入数据为空")
        return {}
    
    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"完成 {i}/{num_merges} 次合并，已用时 {elapsed:.2f} 秒")
        
        stats = get_stats(ids)
        if not stats:
            print(f"警告: 没有更多可合并的对，在 {i}/{num_merges} 次合并后停止")
            break
            
        pair = max(stats, key=stats.get)
        idx = 256 + i
        # print(f"合并 {pair} 为新标记 {idx}，频率: {stats[pair]}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    
    print(f"BPE训练完成，共执行了 {len(merges)} 次合并")
    return merges

def encode(text, merges, vocab):
    """将文本编码为token ID列表"""
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # 找到能合并的最小索引对
        pairs_with_idx = [(p, merges.get(p, float("inf"))) for p in stats.keys()]
        if not pairs_with_idx:
            break
        pair, _ = min(pairs_with_idx, key=lambda x: x[1])
        if pair not in merges:
            break  # 没有更多可合并的内容
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def decode(ids, vocab):
    """将token ID列表解码为文本"""
    if not ids:
        return ""
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

if __name__ == '__main__':
    # Mac路径格式
    folder_path = os.path.expanduser("~/Documents/RAG/Heroes")
    # 如果希望用户输入路径
    # folder_path = input("请输入文件夹路径: ")
    
    vocab_size = 30000
    
    print("开始读取数据...")
    ids = bpe_data(folder_path)
    
    if not ids:
        print("没有读取到数据，程序退出")
        exit(1)
    
    print("开始BPE训练...")
    merges = bpe_func(vocab_size, ids)
    
    # 构建词汇表
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    
    # 保存词汇表和合并规则
    print("保存模型...")
    
    # 测试编码和解码
    test_text = "技能描述"
    encoded = encode(test_text, merges, vocab)
    decoded = decode(encoded, vocab)
    
    print(f"测试文本: '{test_text}'")
    print(f"编码结果: {encoded}")
    print(f"解码结果: '{decoded}'")
    print(f"编码/解码是否一致: {test_text == decoded}")