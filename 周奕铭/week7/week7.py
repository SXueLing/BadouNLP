import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 提取特征和标签
data=pd.read_csv('E:/download/data/practice-7.csv')
X = data['review']
y = data['label']

# 数据预处理：使用TF - IDF向量化文本数据
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 定义模型
models = {
    'Linear SVM': LinearSVC(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression()
}

# 训练和评估模型
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算准确率和分类报告
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': report
    }

# 输出结果
for model_name, result in results.items():
    print(f'Model: {model_name}')
    print(f'Accuracy: {result["accuracy"]}')
    print('Classification Report:')
    print(pd.DataFrame(result['classification_report']).transpose())
    print('-' * 50)
