
import os

def read_files(folder):
    """读取文件夹内所有文件内容，返回字节列表"""
    all_bytes = []
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), 'rb') as f:
            content = f.read()
            all_bytes.extend(list(content))  # 将字节流转为数字列表
    return [all_bytes]  # 用二维列表保存（兼容后续处理）

def count_pairs(bytes_list):
    """统计相邻字节对出现次数"""
    counts = {}
    for i in range(len(bytes_list)-1):
        pair = (bytes_list[i], bytes_list[i+1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_best_pair(bytes_list, pair, new_id):
    """合并最高频字节对"""
    result = []
    i = 0
    while i < len(bytes_list):
        # 查找需要合并的字节对
        if i < len(bytes_list)-1 and bytes_list[i] == pair[0] and bytes_list[i+1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(bytes_list[i])
            i += 1
    return result

def train_bpe(folder, num_merges=10):
    """训练BPE模型"""
    data = read_files(folder)[0]  # 取第一个文件内容
    merges = {}  # 保存合并规则：{(a,b): new_id}
    
    for i in range(num_merges):
        # 1. 统计当前字节对频率
        pair_counts = count_pairs(data)
        if not pair_counts:
            break
        
        # 2. 找到最高频字节对
        best_pair = max(pair_counts, key=pair_counts.get)
        new_id = 256 + i  # 用256以上的数字表示新符号
        
        # 3. 执行合并
        data = merge_best_pair(data, best_pair, new_id)
        
        # 4. 保存规则
        merges[best_pair] = new_id
        print(f"第{i+1}次合并：{bytes(best_pair).decode('utf-8', 'replace')} → {new_id}")
    
    return merges

def encode(text, merges):
    """编码：文本转BPE符号"""
    bytes_list = list(text.encode('utf-8'))
    for (a, b), new_id in merges.items():
        new_bytes = []
        i = 0
        while i < len(bytes_list):
            if i < len(bytes_list)-1 and bytes_list[i] == a and bytes_list[i+1] == b:
                new_bytes.append(new_id)
                i += 2
            else:
                new_bytes.append(bytes_list[i])
                i += 1
        bytes_list = new_bytes
    return bytes_list

def decode(ids, merges):
    """解码：BPE符号转文本"""
    {new_id: (a,b)}
    reverse = {v: k for k, v in merges.items()}
    
    # 逐步替换合并符号
    current = list(ids)
    while True:
        found = False
        for i in range(len(current)):
            if current[i] in reverse:
                # 替换为原始字节对
                a, b = reverse[current[i]]
                current = current[:i] + [a, b] + current[i+1:]
                found = True
                break
        if not found:
            break
    
    # 转换为可读文本
    return bytes(current).decode('utf-8', 'replace')

if __name__ == "__main__":
    merge_rules = train_bpe("../corpus", num_merges=50)   

    text = "矮人直升机"
    encoded = encode(text, merge_rules)
    decoded = decode(encoded, merge_rules)
    print("编码结果:", encoded)
    print("解码结果:", decoded)
