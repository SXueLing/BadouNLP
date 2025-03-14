import os
import jieba
from rank_bm25 import BM25Okapi
from zhipuai import ZhipuAI
import pdfplumber

def call_large_model(prompt):
    client = ZhipuAI(api_key="###") # 填写APIKey
    response = client.chat.completions.create(
        model="glm-4-plus",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content
    return response_text

class SimpleRAG:
    def __init__(self, folder_path="datasets", chunk_size=3):
        self.chunk_size = chunk_size  # 控制返回结果数量
        self.load_title_data(folder_path)
    
    def load_title_data(self, folder_path):
        self.title_data = {}
        self.corpus = []      # 存储分词后的文本块
        self.chunk_info = []  # 存储每个块的元信息

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".txt"):
                self._process_text_file(file_path, file_name)
            elif file_name.endswith(".pdf"):
                self._process_pdf_file(file_path, file_name)
            
        # 初始化BM25模型
        self.bm25_model = BM25Okapi(self.corpus)

    def _process_text_file(self, file_path, file_name):
        """处理文本文件"""
        title = file_name.split(".")[0]
        with open(file_path, "r", encoding="utf-8") as file:
            full_text = file.read()
            self._add_document_content(title, full_text)

    def _process_pdf_file(self, file_path, file_name):
        """处理PDF文件"""
        title = file_name.split(".")[0]
        full_text = ""
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # 提取文本并清理空白字符
                    page_text = page.extract_text().strip()
                    full_text += page_text
        except Exception as e:
            print(f"解析PDF失败：{file_name} - {str(e)}")
            return
        
        self._add_document_content(title, full_text)

    def _add_document_content(self, title, full_text):
        """分块方法"""
        # 按段落初步分割
        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        
        for para in paragraphs:
            # 判断是否为标题行
            is_title = (
                len(para) < 20 and  # 短文本
                not para.endswith('。') and  # 无句号结尾
                any(c.isdigit() for c in para)  # 包含数字编号
            )
            
            if is_title:
                # 遇到标题时完成当前块
                if current_chunk:
                    chunks.append("。".join(current_chunk))
                current_chunk = [para]  # 新块以标题开始
            else:
                # 普通段落添加到当前块
                sentences = [s.strip() for s in para.split('。') if s.strip()]
                current_chunk.extend(sentences)
                
                # 块长度控制（以句子为单位）
                if len(current_chunk) >= 30:
                    chunks.append("。".join(current_chunk))
                    current_chunk = current_chunk[-1:]  # 保留最后一句作为重叠

        # 处理剩余内容
        if current_chunk:
            chunks.append("。".join(current_chunk))

        # 添加到语料库
        for chunk in chunks:
            self.chunk_info.append({
                "title": title,
                "text": chunk,
                "tokens": jieba.lcut(chunk)
            })
            self.corpus.append(self.chunk_info[-1]["tokens"])

    def retrive(self, user_query, top_k=3):
        tokenized_query = jieba.lcut(user_query)
        doc_scores = self.bm25_model.get_scores(tokenized_query)
        
        # 按分数排序，取前top_k个索引
        sorted_indices = sorted(
            range(len(doc_scores)),
            key=lambda i: doc_scores[i],
            reverse=True
        )[:top_k]
        
        # 提取相关信息并去重
        results = []
        seen_titles = set()
        for idx in sorted_indices:
            chunk = self.chunk_info[idx]
            # 确保每个文本最多显示一个最佳结果
            if chunk["title"] not in seen_titles:
                results.append({
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "score": doc_scores[idx]
                })
                seen_titles.add(chunk["title"])
        
        return results[:self.chunk_size]  # 返回指定数量的结果

    def query(self, user_query):    
        print(f"用户问题：{user_query}")
        print("=======================")
        
        results = self.retrive(user_query)
        for i, res in enumerate(results, 1):
            print(f"相关结果 {i}:")
            print(f"标题：{res['title']}")
            print(f"相关文本：{res['text']}")
            print(f"匹配分数：{res['score']:.2f}")
            print("-----------------------")
        
        # 可选：拼接所有相关文本供大模型使用
        context = "\n".join([f"{res['title']}的资料：{res['text']}" for res in results])
        prompt = f"根据以下信息回答问题：\n{context}\n\n问题：{user_query}"
        print(call_large_model(prompt))

if __name__ == "__main__":
    rag = SimpleRAG(chunk_size=5)  # 设置最多返回2个不同文本的相关结果
    rag.query("基础的渗透思路是什么")
