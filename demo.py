import time
import traceback
import os
from openai import OpenAI
import sys

def my_decorator(func):
    def wrapper():
        try:
            return func()
        except Exception as e:
            # 捕获异常的详细信息
            error_str = "Error file: {}\nError line number: {}\nError function: {}\nError code: {}\nError message: {}\n"

            # 获取错误堆栈信息
            error_info = traceback.extract_tb(e.__traceback__)  # 获取堆栈信息
            file, line, func_name, code = error_info[-1]  # 获取最后一个堆栈信息（即当前的错误）

            # 获取异常的具体信息（类型和消息）
            exception_type = type(e).__name__  # 获取异常类型
            exception_message = str(e)  # 获取异常消息

            # 格式化错误信息
            error_str = error_str.format(file, line, func_name, code, exception_message)

            # 打印错误信息
            print(f"{exception_type}: {exception_message}")
            print(error_str)  # 详细错误信息

            return error_str

    return wrapper


def my_decorator2(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_info="Error:\n"
            for i, tb in enumerate(traceback.extract_tb(exc_traceback)[::-1]):
                Traceback=f"Traceback {i+1}:\n"
                File=f"  File: {tb.filename}, line {tb.lineno}, in {tb.name}\n"
                line =f"    {tb.line.strip()}\n"
                error_info+=Traceback+File+File+line
            error_info=error_info+f"错误类型: {exc_type.__name__}"
            return error_info
    return wrapper

def test_model():
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-42b231cc1a5747b6bf0cf32c6df38481",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': '你是谁？你告诉我名字后，请讲一个800字的童话故事给我'}],
        stream=True,
        # stream_options={"include_usage": True}
        )
    return completion


def take_info(completion):
    for chunk in completion:
        print("www",chunk)
        if hasattr(chunk, "choices"):
            data=chunk.choices[0].delta.content
            print("hhh",data)
            if data:
                print("hhh")
                yield data
@my_decorator2
def ouput1():
    out=take_info(test_model())
    return out


if __name__ == '__main__':
    for i in ouput1():
        print(i,end="",flush=True)
































