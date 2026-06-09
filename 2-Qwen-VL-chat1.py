"""
车险VLM智能识别系统 - 单图测试与多轮对话脚本
用于快速测试单张图片分析、多轮对话交互等功能
"""
from config import create_client, parse_image_urls, call_vlm_with_retry, DEFAULT_MODEL


def single_image_test(image_url, prompt="这是什么"):
    """
    单图分析测试
    :param image_url: 图片URL或OSS图片名称
    :param prompt: 分析提示语，默认为"这是什么"
    :return: VLM模型的响应内容
    """
    client = create_client()
    url_list = parse_image_urls(image_url)

    content = [{"type": "text", "text": prompt}]
    for url in url_list:
        content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [{"role": "user", "content": content}]
    completion = call_vlm_with_retry(client, messages)
    result = completion.choices[0].message.content
    print(f"分析结果: {result}")
    return result


def multi_turn_test():
    """
    多轮对话测试示例
    演示如何通过追加assistant和user消息实现多轮对话
    :return: 最后一轮的VLM响应内容
    """
    client = create_client()

    # 第一轮：识别轮毂位置
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "框出图中轮毂的位置"},
                {"type": "image_url", "image_url": {"url": "https://easycar.oss-cn-beijing.aliyuncs.com/car_undistorted.jpg"}},
            ],
        }
    ]
    completion = call_vlm_with_retry(client, messages, model="qwen-vl-plus")
    first_response = completion.choices[0].message.content
    print(f"第一轮结果: {first_response}")

    # 第二轮：基于第一轮结果继续对话
    messages.append({"role": "assistant", "content": first_response})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "图中轮毂的位置在哪里"},
                {"type": "image_url", "image_url": {"url": "https://easycar.oss-cn-beijing.aliyuncs.com/car_undistorted.jpg"}},
            ],
        }
    )
    completion = call_vlm_with_retry(client, messages, model="qwen-vl-plus")
    second_response = completion.choices[0].message.content
    print(f"第二轮结果: {second_response}")
    return second_response


if __name__ == "__main__":
    # 单图测试
    print("=== 单图测试 ===")
    single_image_test(
        "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
        "这是什么",
    )

    # 多轮对话测试
    print("\n=== 多轮对话测试 ===")
    multi_turn_test()
