def read_csv(file_path, rate):
    df = pd.read_csv(file_path)
    train_size = int(df.shape[0] * rate)
    # 按行读取训练集和测试集并写入json文件
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    train_data.to_json('./train.json', orient='records', lines=True, force_ascii=False)
    test_data.to_json('./test.json', orient='records', lines=True, force_ascii=False)
    # 分析正负样本数, 文本平均长度
    ps = 0
    ns = 0  
    avg_len = 0
    for index, row in df.iterrows():
        dic = row.to_dict()
        if dic['label'] == 1:
            ps += 1
        else:
            ns += 1
        avg_len += len(row['review'])   
    avg_len = int(avg_len / df.shape[0])
    print("正样本数：", ps)
    print("负样本数：", ns)
    print("平均文本长度：", avg_len)
