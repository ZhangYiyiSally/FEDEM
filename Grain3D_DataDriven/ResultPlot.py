import numpy as np
import matplotlib.pyplot as plt
import Config as cfg
from Network import ResNet
import torch
import Utility as util
import meshio

def read_error(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # 跳过以%开头的注释行
            if not line.strip().startswith('%'):
                # 处理数据行，按空格分割
                data = [float(val) for val in line.strip().split()]
    return data

Uer_Num0=[]
Uer_Num2=[]
Uer_Num5=[]
Uer_Num10=[]
Uer_Num20=[]
Ser_Num0=[]
Ser_Num2=[]
Ser_Num5=[]
Ser_Num10=[]
Ser_Num20=[]

mesh = meshio.read(cfg.mesh_path, file_format="gmsh")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 绘制训练步骤
range_x=100000
steps = np.arange(10000, range_x+1, 10000)

Uer_Num0=read_error("Grain3D/Results_Adam/e4_mesh10828_Net4x3-200-3_Exp0.0005_weight100000.0/int2/U_error.txt")[1:]
Uer_Num2=read_error("Grain3D_DataDriven/Results_Adam/DrivenNum2/U_error.txt")[1:]
Uer_Num5=read_error("Grain3D_DataDriven/Results_Adam/DrivenNum5/U_error.txt")[1:]
Uer_Num10=read_error("Grain3D_DataDriven/Results_Adam/DrivenNum10/U_error.txt")[1:]
Uer_Num20=read_error("Grain3D_DataDriven/Results_Adam/DrivenNum20/U_error.txt")[1:]
Ser_Num0=read_error("Grain3D/Results_Adam/e4_mesh10828_Net4x3-200-3_Exp0.0005_weight100000.0/int2/S_error.txt")[1:]
Ser_Num2=read_error("Grain3D_DataDriven/Results_Adam/DrivenNum2/S_error.txt")[1:]
Ser_Num5=read_error("Grain3D_DataDriven/Results_Adam/DrivenNum5/S_error.txt")[1:]
Ser_Num10=read_error("Grain3D_DataDriven/Results_Adam/DrivenNum10/S_error.txt")[1:]
Ser_Num20=read_error("Grain3D_DataDriven/Results_Adam/DrivenNum20/S_error.txt")[1:]

# 创建图形和轴
plt.figure(1,figsize=(15, 10))  # 设置图形大小

# 绘制曲线
plt.plot(steps, Uer_Num0, label='No driving data', color='black', linewidth=2, marker='o', markersize=6, linestyle='-')
plt.plot(steps, Uer_Num2, label='2 driving data', color='red', linewidth=2, marker='o', markersize=6, linestyle='-')
plt.plot(steps, Uer_Num5, label='5 driving data', color='blue', linewidth=2, marker='o', markersize=6, linestyle='-')
plt.plot(steps, Uer_Num10, label='10 driving data', color='green', linewidth=2, marker='o', markersize=6, linestyle='-')
plt.plot(steps, Uer_Num20, label='20 driving data', color='purple', linewidth=2, marker='o', markersize=6, linestyle='-')

# 设置x轴范围
plt.xlim(10000, 100000)
# 添加标题和标签
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('Mean relative error of displacement', fontsize=16)
# 调整坐标刻度的字体大小
plt.tick_params(axis='both', which='major', labelsize=14)
# 添加图例
plt.legend(fontsize=14, loc='upper left')
# 添加网格
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# 创建图形和轴
plt.figure(2,figsize=(15, 10))  # 设置图形大小

# 绘制曲线
plt.plot(steps, Ser_Num0, label='No driving data', color='black', linewidth=2, marker='o', markersize=6, linestyle='-')
plt.plot(steps, Ser_Num2, label='2 driving data', color='red', linewidth=2, marker='o', markersize=6, linestyle='-')
plt.plot(steps, Ser_Num5, label='5 driving data', color='blue', linewidth=2, marker='o', markersize=6, linestyle='-')
plt.plot(steps, Ser_Num10, label='10 driving data', color='green', linewidth=2, marker='o', markersize=6, linestyle='-')
plt.plot(steps, Ser_Num20, label='20 driving data', color='purple', linewidth=2, marker='o', markersize=6, linestyle='-')

# 设置x轴范围
plt.xlim(10000, 100000)
# 添加标题和标签
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('Mean relative error of stress', fontsize=16)
# 调整坐标刻度的字体大小
plt.tick_params(axis='both', which='major', labelsize=14)
# 添加图例
plt.legend(fontsize=14, loc='upper left')
# 添加网格
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# 显示图形
plt.show()