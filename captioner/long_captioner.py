import os
from openai import OpenAI
import glob 
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


for image_path in glob.glob("../data/rename_structured3d/*.png"):
    print(image_path)
    base64_image = encode_image(image_path)
    print("done encoding")
    completion = client.chat.completions.create(
        # model="qwen-plus", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        # model="qwen-vl-max-2025-01-25",
        model="qwen-vl-plus-2025-01-25",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 需要注意，传入BASE64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        "min_pixels": 28 * 28 * 4,
                        "max_pixels": 28 * 28 * 1280
                    },
                    # 为保证识别效果，目前模型内部会统一使用"Read all the text in the image."进行识别，用户输入的文本不会生效。
                    {"type": "text", "text": "Describe the image in detail. Start with 'A 360-degree panoramic image of a ...'"},
                ],
            } 
        ]
    ) 
    print(completion.choices[0].message.content)
    raise