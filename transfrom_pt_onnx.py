import torch
import onnxruntime as ort
import onnx
import os
# 导入自己定义的模型，注意这个模型一定要放在main主函数的上面，不然导入的时候，会直接把整个py文件给执行了
from my_resnet import MyResnet18


# 有GPU就用GPU，没有就用CPU
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 自己的模型,通过这种方式加载模型，需要将自己训练的模型导入到这个py文件，不然会报下面的错
# 并且保存模型的时候是使用torch.save(model, "best_model1.pt")这种方式
# 报错信息：AttributeError: Can't get attribute 'MyResnet18' on <module '__main__' from 'E:/my_net/transfrom_pt_onnx.py'>
model=torch.load("best_model1.pt")

# # 保存方式为torch.save(model.state_dict(), "best_model2.pt")
# # 报错：AttributeError: 'collections.OrderedDict' object has no attribute 'eval'
# model=MyResnet18()
# model.load_state_dict(torch.load("best_model2.pt")) # 这里一定不要赋值给model，否则会报错
# model=model.to(devices)

model.eval()
# 为什么export需要一个输入呢？
# 因为转出onnx本质就是一种语言的翻译，从一种格式转出另一种。torch提供了一种追踪机制，给一个输入，让模型在执行一遍
# 将这个输入对应的计算图记录下来并保存成onnx
x= torch.randn(1, 3, 224, 224).to(devices)
with torch.no_grad():
    ort_model = torch.onnx.export(
        model,  # 要转换的模型
        x,  # 任意一组输入(记得是float类型)
        "best_model1.onnx",  # 导出的文件
        input_names=["input"],  # 输入tensor的名称
        output_names=["output"],  # 输出tensor的名称
        opset_version=11)  # onnx算子集版本号

# 读取onnx模型
onnx_model=onnx.load("best_model1.onnx")
# # 检查模型格式是否正确
onnx.checker.check_model(onnx_model)
print("onnx模型格式正确!")














