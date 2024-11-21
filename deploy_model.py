import onnxruntime as ort
import torch
import cv2
from PIL import Image
from torchvision import transforms

# 选择GPU还是CPU
provider = ort.get_available_providers()[1 if ort.get_device() == 'GPU' else 0]
# 载入onnx模型
onnx_model = ort.InferenceSession("best_model1.onnx")

# 构造随机输入，获取结果看是否正确
x = torch.randn(1, 3, 224, 224).numpy()

#  输出是一个列表，里面只有一个元素，元素类型是2维数组元素，存着每个分类的概率
# [array([[-0.09820878, -1.3899375 ,  1.5437527 , -0.5629988 ,  0.10385571]],dtype=float32)]
model_output = onnx_model.run(["output"], {"input": x})[0]

# 图像预处理,图像预处理方式要和模型训练时候的预处理方式保持一致
data_preproce = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_cv = cv2.imread("test.png")
img_pil = Image.fromarray(img_cv)
img = data_preproce(img_pil)
input_tensor = torch.unsqueeze(img, 0).numpy()
# print(input_tensor.shape) # (1, 3, 224, 224)
# {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}

# 对预测的值进行softmax，得到置信度
pred = onnx_model.run(["output"], {"input": input_tensor})[0]
# print(pred.shape) # (1, 5)
pred_softmax = torch.softmax(torch.tensor(pred), dim=1)
# 置信度前三个结果
n = 3
values, indices = torch.topk(pred_softmax, n)  # 返回的是结果，以及这个结果所在原来tensor中的索引位置
# 预测类别
index_n = indices.tolist()[0]
values_n = values.tolist()[0]

src = cv2.imread("test.png")
labels_dict = {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
revers_dict = {v: k for k, v in labels_dict.items()}
position = 30
for i in index_n:
    print(revers_dict[i], ":", round(values_n[index_n.index(i)] * 100, 5), "%")
    # 在原图的左上角显示前三个类别的概率
    cv2.putText(src,
                revers_dict[i] + " : " + str(round(values_n[index_n.index(i)] * 100, 5)) + "%",
                (10, position),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    position += 30

cv2.imshow("src", src)
cv2.imwrite("result.png", src)
cv2.waitKey()
cv2.destroyAllWindows()
