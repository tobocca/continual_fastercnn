import torch

class Model(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(Model, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        return x


# model = Model(10,20,10)
# x = torch.randn(5,10).requires_grad_(True)  # 定义一个网络的输入值
# y = model(x)  # 获取网络的预测值
# MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# MyConvNetVis.format = "png"
# # 指定文件生成的文件夹
# MyConvNetVis.directory = "./zhq_vis/"
# # 生成文件
# MyConvNetVis.view()

import hiddenlayer as h
import torchvision

model = torchvision.models.resnet18()
vis_graph = h.build_graph(model, torch.zeros([1, 3, 128, 128]))  # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
vis_graph.save("zhq_vis/demo1.png")  # 保存图像的路径
