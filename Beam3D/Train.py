import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import time
import meshio
import Config as cfg
from Network import ResNet
from Dataset import Dataset
from Loss import Loss
import Utility as util

def plot_loss(loss, error):
    # 绘制损失曲线
    # plt.clf()
    # plt.figure(1)
    plt.plot(loss, linewidth=2, color='firebrick')
    plt.plot(error, linewidth=2, color='blue')
    plt.tick_params(axis='both', size=5, width=2,
                    direction='in', labelsize=15)
    plt.xlabel('Iteration', size=15)
    plt.ylabel('Loss', size=15)
    plt.legend(['Loss', 'errorL2'], loc='upper right', fontsize=15)
    plt.title('Training Curve', size=20)
    plt.grid(color='midnightblue', linestyle='-.', linewidth=0.5)
    # 调整坐标轴的边框样式
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)  # 设置边框宽度为2
    plt.pause(0.0001)


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    mesh = meshio.read(cfg.mesh_path, file_format="gmsh")
    data=Dataset()
    dom=data.domain(mesh)
    print("加载网络：%s" % cfg.mesh_path)

    bc_Dir=data.bc_Dirichlet('bc_Dirichlet')
    bc_Neu=data.bc_Neumann('bc_Neumann')

    # 定义神经网络，神经网络的输入为空间坐标，输出为三个方向的位移
    dem=ResNet(cfg.input_size, cfg.hidden_size, cfg.output_size, cfg.depth).to(dev)
    dem.train()

    # 开始训练, 设置训练参数
    start_time = time.time()
    losses = []
    eL2=[]
    eH1=[]
    epoch_num=cfg.epoch_num
    learning_rate_AdamW=cfg.learning_rate_AdamW
    learning_rate_LBFGS=cfg.learning_rate_LBFGS
    max_iter_LBFGS=cfg.max_iter_LBFGS
        
    # 定义优化器
    optimizer_AdamW = torch.optim.AdamW(dem.parameters(), lr=learning_rate_AdamW, foreach=True)
    optimizer_LBFGS = torch.optim.LBFGS(dem.parameters(), lr=learning_rate_LBFGS, max_iter=max_iter_LBFGS)

    print(f"开始训练：共{epoch_num}个epoch，学习率{learning_rate_LBFGS}")
    tqdm_epoch = tqdm(range(epoch_num), desc='epoches',colour='red', dynamic_ncols=True)
    for epoch in range(epoch_num):

        # # 计算损失函数
        # loss=Loss(dem)
        # loss_value=loss.loss_function(dom, bc_Dir, bc_Neu)
        # # losses.append(loss_value.item())
        # eL2.append(util.errorL2(dem, mesh.points, dev).item())
        # eH1.append(util.errorH1(dem, mesh.points, dev).item())
        # # 反向传播
        # optimizer_AdamW.zero_grad()
        # loss_value.backward()
        # optimizer_AdamW.step()

        def closure():
            loss=Loss(dem)
            loss_closure=loss.loss_function(dom, bc_Dir, bc_Neu)
            losses.append(loss_closure.item())
            eL2.append(util.errorL2(dem, mesh.points, dev).item())
            eH1.append(util.errorH1(dem, mesh.points, dev).item())
            # 反向传播
            optimizer_LBFGS.zero_grad()
            loss_closure.backward()
            return loss_closure
            
            
        optimizer_LBFGS.step(closure=closure)


        # 更新epoch进度条
        tqdm_epoch.update()
        tqdm_epoch.set_postfix(errorL2='{:.5f}'.format(eL2[-1]))
        plot_loss(losses, eL2)


    # 保存模型
    os.makedirs(cfg.model_save_path, exist_ok=True)
    torch.save(dem.state_dict(), f"{cfg.model_save_path}/dem.pth")
    plt.savefig(f"{cfg.model_save_path}/training_curve_lr{learning_rate_LBFGS}.png")
    with open(f"{cfg.model_save_path}/loss_seed{cfg.seed}.txt", 'w') as f:
        f.write('\n'.join(map(str, losses)) + '\n')
    with open(f"{cfg.model_save_path}/errorL2_seed{cfg.seed}.txt", 'w') as f:
        f.write('\n'.join(map(str, eL2)) + '\n')
    with open(f"{cfg.model_save_path}/errorH1_seed{cfg.seed}.txt", 'w') as f:
        f.write('\n'.join(map(str, eH1)) + '\n')

    end_time=time.time()-start_time
    print("End time: %.5f" % end_time)
    print("训练结束：结果保存在%s" % cfg.model_save_path)