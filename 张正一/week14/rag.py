import env
import os
from openai import OpenAI
print(type(os.environ.get("KIMI_TOKEN")))
client = OpenAI(
  api_key=f'{os.environ.get("KIMI_TOKEN")}',
  base_url = "https://api.moonshot.cn/v1",
)

completion = client.chat.completions.create(
    model="moonshot-v1-8k", # 指定特定模型
    messages=[ 
        # 也可以不设置system字段
        {'role': 'system', 'content': '平台助手'},
        {'role': 'user', 'content': '你好'}
    ]
)

print(completion.choices[0].message.content)