import torch
import torch.nn as nn


class ResidualBlock(nn.Module):  # 残差块
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.activate = nn.Tanh()  # 激活函数

         # 初始化参数
        torch.nn.init.normal_(self.fc1.bias, mean=0, std=0.1)
        torch.nn.init.normal_(self.fc2.bias, mean=0, std=0.1)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=0.1)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activate(out)

        out = self.fc2(out)
        out += residual  # 添加残差连接
        out = self.activate(out)  # 激活函数
        return out

class ResNet(nn.Module):  # 残差神经网络
    def __init__(self, input_size: int, hidden_size: int, output_size: int, depth: int):
        super(ResNet, self).__init__()
        self.layers = nn.ModuleList()  # 使用 ModuleList 来存储所有的层
        self.layers.append(nn.Linear(input_size, hidden_size))  # 添加输入层

        # 初始化输入层的参数
        torch.nn.init.normal_(self.layers[0].bias, mean=0, std=0.001)
        torch.nn.init.normal_(self.layers[0].weight, mean=0, std=0.001)

        for i in range(depth - 1):  # 添加残差块
            self.layers.append(ResidualBlock(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, output_size))  # 添加输出层

        # 初始化输出层的参数
        torch.nn.init.normal_(self.layers[-1].bias, mean=0, std=0.001)
        torch.nn.init.normal_(self.layers[-1].weight, mean=0, std=0.001)
        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    # 创建神经网络
    model = ResNet(input_size=3, hidden_size=128, output_size=1, depth=3)
    input=torch.tensor([[[[1.0, 2.0 ,3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]],
                        [[[1.0, 2.0 ,3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]],
                        [[[1.0, 2.0 ,3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]],
                        [[[1.0, 2.0 ,3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]])
    output=model(input)
    print(output)
