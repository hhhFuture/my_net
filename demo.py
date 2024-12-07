import time
import traceback

def my_decorator(func):
    def wrapper():
        try:
            return func()
        except Exception as e:
            error_str="Error file：{}\nError line number：{}\nError funtion：{}\nError code：{}"
            error_info =traceback.extract_tb(e.__traceback__)
            for file,line,func_name,code in error_info:
                error_str=error_str.format(file,line,func_name,code)
            return error_str
    return wrapper

def test(func):
    try:
        return func()
    except Exception as e:
        error_str = "Error file：{}\nError line number：{}\nError funtion：{}\nError code：{}"
        error_info = traceback.extract_tb(e.__traceback__)
        for file, line, func_name, code in error_info:
            error_str = error_str.format(file, line, func_name, code)
        return error_str
# 使用装饰器








# def test():
#     try:
#         a = 1 / 0  # 会产生ZeroDivisionError
#     except Exception as e:
#         # 使用extract_tb获取信息，不调用print_exc()
#         tb_info = traceback.extract_tb(e.__traceback__)
#         for filename, lineno, funcname, text in tb_info:
#             print(f"Error file：{filename}")
#             print(f"Error line number：{lineno}")
#             print(f"Error funtion：{funcname}")
#             print(f"Error code：{text}")
#
# test()




























