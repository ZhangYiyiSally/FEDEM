#---------------------------------随机种子-----------------------------------------
seed = 2025
#---------------------------------网格设置-----------------------------------------
data_num=5  # 训练数据集数量
#--------------------------------高斯积分精度设置-----------------------------------------
n_int3D=2  # 三维高斯积分精度
n_int2D=2  # 二维高斯积分精度

#-------------------------------材料参数设置-------------------------------------
# ----------E：杨氏模量，nu：泊松比---------------
E=20e6  # 单位：Pa
nu=0.49  # 泊松比
#-------------------------------Dirichlet边界条件设置-------------------------------------
# ----------Dir_marker：边界标记，Dir_u：边界指定位移---------------
Dir_marker='OutSurface'
Dir_u=[0.0, 0.0, 0.0]
#-------------------------------压力边界条件设置-------------------------------------
# ----------Pre_marker：边界标记，Pre_value：边界指定力---------------
Pre_marker='InSurface'
Pre_value=1e6  # 单位：Pa
#-------------------------------对称边界条件设置-------------------------------------
# ----------Sym_marker：边界标记---------------
Sym_marker='Symmetry'
#--------------------------------神经网络设置-----------------------------------------
input_size=3  # ResNet的输入大小
hidden_size=256  # ResNet的隐藏层大小
output_size=3  # ResNet的输出大小
depth=4  # ResNet的深度
latent_dim=128  # 隐变量的维度
#--------------------------------训练参数设置-----------------------------------------
epoch_num=100000  # 训练的epoch数
lr=5e-4 #  学习率
lr_scheduler='Exp'
loss_weight=1e5 # 边界损失函数的权重

#---------------------------文件路径-----------------------------------------------
mesh_path=f"GroupGrains/models"
model_save_path=f"GroupGrains/Results_Adam/DataNum{data_num}/Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr}_weight{loss_weight}"
Evaluate_save_path=f"GroupGrains/Results_Adam/DataNum{data_num}/Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr}_weight{loss_weight}"