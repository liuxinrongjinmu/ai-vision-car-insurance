import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxxx", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-vl-max", #qwen-vl-plus
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是什么"},
                {"type": "image_url","image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}}
            ]
        }
    ]
)
print(completion.model_dump_json())

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "框出图中轮毂的位置"},
            {"type": "image_url","image_url": {"url": "https://easycar.oss-cn-beijing.aliyuncs.com/car_undistorted.jpg"}}
        ]
    }
]

completion = client.chat.completions.create(
    model="qwen-vl-max-2024-08-09",
    messages=messages
)
print(completion.model_dump_json())

help(completion)

messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
messages.append({
    "role": "user",
    "content": [
        {"type": "text", "text": "图中轮毂的位置在哪里"},
        {"type": "image_url","image_url": {"url": "https://easycar.oss-cn-beijing.aliyuncs.com/car_undistorted.jpg"}}
    ]
})
print(messages)

completion = client.chat.completions.create(
    model="qwen-vl-plus",
    messages=messages
)
print(completion.model_dump_json())

print(completion)

# 以下部分为 tokenizer 和 model 的多轮对话及图片处理示例，需确保相关库和模型已正确加载
# query = tokenizer.from_list_format([
#     {'image': 'https://easycar.oss-cn-beijing.aliyuncs.com/2.jpg'},
#     {'text': '这是什么?'}
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)
# response, history = model.chat(tokenizer, '框出图中轮毂的位置', history=history)
# print(response)
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#     image.save('wheel.jpg')
# else:
#     print("no box")

# response, history = model.chat(tokenizer, '框出图中凹陷和划痕的位置', history=history)
# print(response)
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#     image.save('car_damage.jpg')
# else:
#     print("no box")

import gradio
print(gradio.__version__)
import transformers
print(transformers.__version__)
import torch
print(torch.__version__)

# matplotlib 图像展示示例
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('2.jpg')
plt.imshow(img)
plt.title('2.jpg')
plt.axis('off')
plt.show()

img = mpimg.imread('3.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()

# tokenizer 多轮对话示例
# query = tokenizer.from_list_format([
#     {'image': 'https://vl-image.oss-cn-shanghai.aliyuncs.com/3.jpg'},
#     {'text': '厦门航空在哪个区？'}
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)
# query = tokenizer.from_list_format([
#     {'image': 'https://vl-image.oss-cn-shanghai.aliyuncs.com/3.jpg'},
#     {'text': 'A区有哪些航空公司？'}
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)

img = mpimg.imread('4.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()

# query = tokenizer.from_list_format([
#     {'image': 'https://vl-image.oss-cn-shanghai.aliyuncs.com/4.jpg'},
#     {'text': '机场巴士在第几层？'}
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)

img = mpimg.imread('5.png')
plt.imshow(img)
plt.axis('off')
plt.show()

# query = tokenizer.from_list_format([
#     {'image': 'https://vl-image.oss-cn-shanghai.aliyuncs.com/5.png'},
#     {'text': '头等舱免费行李额是多少KG？'}
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)
