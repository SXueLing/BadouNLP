from zhipuai import ZhipuAI
client = ZhipuAI(api_key="4c675a61a94347f9bd4a7c6803fc594b.KTpJsKkFO3Lduwfr")  # 请填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4-plus",  # 请填写您要调用的模型名称
    messages=[
        {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的口号"},
    ],
)
print(response.choices[0].message)