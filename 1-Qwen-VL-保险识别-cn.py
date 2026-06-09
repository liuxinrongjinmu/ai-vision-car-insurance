"""
车险VLM智能识别系统 - 批量分析主脚本
读取Excel中的分析模板，批量调用Qwen-VL模型进行车险图像分析，将结果保存到Excel
"""
import os
from datetime import datetime
import pandas as pd
from config import create_client, parse_image_urls, call_vlm_with_retry


def get_response(client, user_prompt, image_url_str):
    """
    调用VLM模型获取图像分析结果
    :param client: OpenAI客户端实例
    :param user_prompt: 用户分析提示语
    :param image_url_str: 图片名称字符串，支持单图或多图列表格式"[name1, name2]"
    :return: VLM模型的响应结果
    """
    # 解析图片URL列表
    image_url_list = parse_image_urls(image_url_str)

    # 构造消息内容
    content = [{"type": "text", "text": user_prompt}]
    for url in image_url_list:
        content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [{"role": "user", "content": content}]

    print(f"  图片数量: {len(image_url_list)}, 提示语: {user_prompt[:30]}...")

    # 带重试的API调用
    completion = call_vlm_with_retry(client, messages)
    return completion


def batch_analyze(input_excel, output_excel=None):
    """
    批量分析Excel中的车险图像任务
    :param input_excel: 输入Excel文件路径，需包含prompt和image列
    :param output_excel: 输出Excel文件路径，默认自动生成带日期的文件名
    :return: 包含分析结果的DataFrame
    """
    # 生成输出文件名
    if output_excel is None:
        date_str = datetime.now().strftime("%Y%m%d")
        output_excel = f"./prompt_template_cn_result-{date_str}.xlsx"

    # 创建API客户端
    client = create_client()

    # 读取输入数据
    df = pd.read_excel(input_excel)
    df["response"] = ""

    total = len(df)
    success_count = 0
    fail_count = 0

    print(f"开始批量分析，共 {total} 条任务")
    print("-" * 50)

    for index, row in df.iterrows():
        user_prompt = str(row["prompt"])
        image_url = str(row["image"])
        print(f"[{index + 1}/{total}] 处理中...")

        try:
            completion = get_response(client, user_prompt, image_url)
            response = completion.choices[0].message.content
            df.loc[index, "response"] = response
            success_count += 1
            print(f"  成功: {response[:50]}...")
        except Exception as e:
            error_msg = f"分析失败: {e}"
            df.loc[index, "response"] = error_msg
            fail_count += 1
            print(f"  失败: {e}")

        # 每处理5条保存一次中间结果，防止中断丢失
        if (index + 1) % 5 == 0:
            df.to_excel(output_excel, index=False)
            print(f"  [中间保存] 已保存到 {output_excel}")

    # 最终保存
    df.to_excel(output_excel, index=False)

    print("-" * 50)
    print(f"批量分析完成！成功: {success_count}, 失败: {fail_count}")
    print(f"结果已保存到: {output_excel}")
    print(df)
    return df


if __name__ == "__main__":
    batch_analyze("./prompt_template_cn.xlsx")
