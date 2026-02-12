import os
from openai import OpenAI
import subprocess

# 1. 检查当前 Python 看到的变量
print(f"Python 看到的值: {os.getenv('DASHSCOPE_API_KEY')}")

# 2. 模拟 Login Shell 执行，看能不能拿到
result = subprocess.check_output("bash -lc 'echo $DASHSCOPE_API_KEY'", shell=True)
print(f"Login Shell 看到的值: {result.decode().strip()}")

try:
    print(os.getenv("DASHSCOPE_API_KEY"))
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
        # api_key="sk-cb569019acb944b78b829648864d3af8",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
    # completion = client.chat.completions.create(
    #     model="qwen3-max",
    #     messages=[{'role': 'user', 'content': '你是谁？'}]
    # )
    completion = client.chat.completions.create(
        model="qwen-mt-flash",  # 选择模型
        # messages 有且仅有一个 role 为 user 的消息，其 content 为待翻译文本
        messages=[{"role": "user", "content": "我看到这个视频后没有笑"}],
        # 由于 translation_options 非 OpenAI 标准参数，需要通过 extra_body 传入
        extra_body={"translation_options": {"source_lang": "Chinese", "target_lang": "English"}},
    )

    print(completion.choices[0].message.content)
    print(completion.model)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")
