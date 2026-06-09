"""
车险VLM智能识别系统 - 公共配置模块
提供API客户端初始化、图片URL拼接、环境变量读取等公共功能
"""
import os
import ast
import time
from openai import OpenAI

# OSS图片基础URL
OSS_IMAGE_BASE_URL = "https://vl-image.oss-cn-shanghai.aliyuncs.com"

# 默认模型
DEFAULT_MODEL = "qwen-vl-max-2025-04-08"

# API调用间隔（秒），用于速率控制
API_CALL_INTERVAL = 1.0

# 最大重试次数
MAX_RETRIES = 3

# 重试等待时间（秒）
RETRY_WAIT = 5


def get_api_key():
    """
    从环境变量获取DashScope API Key
    :return: API Key字符串
    :raises EnvironmentError: 未配置DASHSCOPE_API_KEY环境变量时抛出
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "未配置DASHSCOPE_API_KEY环境变量。"
            "请执行: set DASHSCOPE_API_KEY=your-api-key (Windows) "
            "或 export DASHSCOPE_API_KEY=your-api-key (Linux/Mac)"
        )
    return api_key


def create_client():
    """
    创建OpenAI兼容模式的DashScope客户端
    :return: OpenAI客户端实例
    """
    return OpenAI(
        api_key=get_api_key(),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def parse_image_urls(image_url_str):
    """
    解析图片URL字符串，支持单图和多图格式
    单图格式: "image_name"
    多图格式: "[image_name1, image_name2, ...]"
    :param image_url_str: 图片名称字符串，支持单个名称或列表格式
    :return: 图片完整URL列表
    """
    image_url_str = image_url_str.strip()

    # 尝试解析为列表格式
    if image_url_str.startswith("["):
        # 先尝试标准Python列表格式: ["img1", "img2"]
        try:
            name_list = ast.literal_eval(image_url_str)
            if isinstance(name_list, list):
                return [build_oss_url(str(name).strip()) for name in name_list]
        except (ValueError, SyntaxError):
            pass
        # 回退：手动解析无引号列表格式: [img1, img2, img3]
        inner = image_url_str[1:].rstrip("]")
        if inner.strip():
            name_list = [name.strip() for name in inner.split(",") if name.strip()]
            if name_list:
                return [build_oss_url(name) for name in name_list]

    # 单图格式
    return [build_oss_url(image_url_str)]


def build_oss_url(image_name):
    """
    根据图片名称拼接OSS完整URL
    :param image_name: 图片文件名（不含基础URL，可含或不含.jpg后缀）
    :return: 完整的OSS图片URL
    """
    image_name = image_name.strip()
    if not image_name.lower().endswith(".jpg") and not image_name.lower().endswith(".png"):
        image_name = f"{image_name}.jpg"
    return f"{OSS_IMAGE_BASE_URL}/{image_name}"


def call_vlm_with_retry(client, messages, model=None):
    """
    带重试机制的VLM调用
    :param client: OpenAI客户端实例
    :param messages: 消息列表
    :param model: 模型名称，默认使用DEFAULT_MODEL
    :return: API响应结果
    :raises RuntimeError: 超过最大重试次数仍失败时抛出
    """
    if model is None:
        model = DEFAULT_MODEL

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            # 速率控制
            time.sleep(API_CALL_INTERVAL)
            return completion
        except Exception as e:
            print(f"  [重试 {attempt}/{MAX_RETRIES}] API调用失败: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT * attempt)
            else:
                raise RuntimeError(f"API调用失败，已重试{MAX_RETRIES}次: {e}")
