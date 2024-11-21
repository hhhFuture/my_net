import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from pokemans import Pokmans_Data
from torch.utils.data import DataLoader
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


class MyResnet18(nn.Module):
    def __init__(self):
        super(MyResnet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.net = nn.Sequential(*list(self.resnet18.children())[:-1],
                                 nn.Flatten(),
                                 nn.Linear(512, 5))

    def forward(self, x):
        return self.net(x)


def evalut(model, dataLoader):
    model.eval()  # 转出评估模式
    acc = 0
    for labels, datas in dataLoader:
        labels, datas = labels.to("cuda"), datas.to("cuda")
        with torch.no_grad():
            # 模型的输出是torch.Size([32, 5])，32张图片，每张图片有5个分类，所以输出是5
            pred = model(datas).argmax(dim=1)
        acc += torch.eq(pred, labels).sum().float().item()
    return acc / len(dataLoader.dataset)


if __name__ == '__main__':

    # epochs = 10
    # model = MyResnet18()
    # # a=torch.randn((32,3,224,224))
    # # print(model(a).shape)#torch.Size([32, 5])
    # criterion = nn.CrossEntropyLoss()
    # model = model.to("cuda")
    # criterion = criterion.to("cuda")
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # trainDatas = Pokmans_Data(r"E:\pokeman", mode="train")
    # testDatas = Pokmans_Data(r"E:\pokeman", mode="test")
    # trainDataLoader = DataLoader(trainDatas, batch_size=32, shuffle=True)
    # testDataLoader = DataLoader(testDatas, batch_size=32, shuffle=True)
    # best_acc, best_epoch = 0, 0
    # epoch_list = []
    # loss_list = []
    #
    # for epoch in range(epochs):
    #     for i, (labels, datas) in enumerate(trainDataLoader):
    #         # 将数据集放到cuda中训练
    #         labels, datas = labels.to("cuda"), datas.to("cuda")
    #
    #         model.train()
    #         loss = criterion(model(datas), labels) # 损失函数
    #         optimizer.zero_grad() # 清空梯度
    #         loss.backward() # 反向传播
    #         optimizer.step() # 优化器更新参数
    #     print("epoch:", epoch, "loss:", loss.item())
    #
    #     # 根据测试集的输出，保存最好的pt模型
    #     if epoch % 1 == 0:
    #         val_acc = evalut(model, testDataLoader)  # 测试集的acc
    #         if val_acc > best_acc:
    #             best_acc, best_epoch = val_acc, epoch
    #             torch.save(model, "best_model1.pt")
    #             torch.save(model.state_dict(), "best_model2.pt")
    #
    #     # 绘制每一轮loss的变换的折线图。
    #     epoch_list.append(epoch)
    #     loss_list.append(loss.item())
    #     best_index = epoch_list.index(best_epoch)
    #     best_loss = loss_list[best_index]
    #     plt.plot(epoch_list, loss_list, color="red", linewidth=2)
    #     plt.title("loss change")
    #     plt.xlabel("epoch")
    #     plt.ylabel("loss")
    #     plt.savefig("loss_change.png")
    #
    # print("best_acc:", best_acc, "best_epoch:", best_epoch, "loss:", best_loss, "val_acc:", val_acc)
    # plt.show()
    # plt.close()

    epochs = 10
    model = MyResnet18().to("cuda")
    net=nn.DataParallel(model,device_ids=[0,1])
    # a=torch.randn((32,3,224,224))
    # print(net(a).shape)#torch.Size([32, 5])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to("cuda")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trainDatas = Pokmans_Data(r"E:\pokeman", mode="train")
    testDatas = Pokmans_Data(r"E:\pokeman", mode="test")
    trainDataLoader = DataLoader(trainDatas, batch_size=32, shuffle=True)
    testDataLoader = DataLoader(testDatas, batch_size=32, shuffle=True)
    best_acc, best_epoch = 0, 0
    epoch_list = []
    loss_list = []

    for epoch in range(epochs):
        for i, (labels, datas) in enumerate(trainDataLoader):
            # 将数据集放到cuda中训练
            labels, datas = labels.to("cuda"), datas.to("cuda")

            net.train()
            optimizer.zero_grad()  # 清空梯度
            loss = criterion(net(datas), labels)  # 损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数
        print("epoch:", epoch, "loss:", loss.item())

        # 根据测试集的输出，保存最好的pt模型
        if epoch % 1 == 0:
            val_acc = evalut(net, testDataLoader)  # 测试集的acc
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                # torch.save(model, "best_model1.pt")
                # torch.save(model.state_dict(), "best_model2.pt")

        # 绘制每一轮loss的变换的折线图。
        epoch_list.append(epoch)
        loss_list.append(loss.item())
        best_index = epoch_list.index(best_epoch)
        best_loss = loss_list[best_index]
        plt.plot(epoch_list, loss_list, color="red", linewidth=2)
        plt.title("loss change")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        # plt.savefig("loss_change.png")

    print("best_acc:", best_acc, "best_epoch:", best_epoch, "loss:", best_loss, "val_acc:", val_acc)
    plt.show()
    plt.close()
