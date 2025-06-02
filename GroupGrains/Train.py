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

def plot_loss(loss, error):
    # 绘制损失曲线
    # plt.clf()
    # plt.figure(1)
    plt.plot(loss, linewidth=2, color='firebrick')
    plt.plot(error, linewidth=2, color='blue')
    plt.tick_params(axis='both', size=5, width=2,
                    direction='in', labelsize=15)
    plt.xlabel('epoch', size=15)
    plt.ylabel('Loss', size=15)
    plt.legend(['Loss', 'lr'], loc='upper right', fontsize=15)
    plt.title('Training Curve', size=20)
    plt.grid(color='midnightblue', linestyle='-.', linewidth=0.5)
    plt.ylim(-2000, 1)
    # 调整坐标轴的边框样式
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)  # 设置边框宽度为2
    plt.pause(0.0001)


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    torch.manual_seed(cfg.seed)

    data=Dataset(data_path=cfg.mesh_path, data_num=cfg.data_num)
    dom=data.domain()
    bc_Dir=data.bc_Dirichlet(cfg.Dir_marker)
    bc_Pre=data.bc_Pressure(cfg.Pre_marker)
    bc_Sym=data.bc_Symmetry(cfg.Sym_marker)

    # 定义神经网络，神经网络的输入为空间坐标，输出为三个方向的位移
    dem=ResNet(cfg.input_size, cfg.hidden_size, cfg.output_size, cfg.depth, cfg.data_num, cfg.latent_dim).to(dev)
    start_epoch=0
    dem.train()

    # 开始训练, 设置训练参数
    start_time = time.time()
    losses = []
    epoch_num=cfg.epoch_num
    learning_rate=cfg.lr
        
    # 定义优化器
    # optimizer = torch.optim.LBFGS(dem.parameters(), lr=learning_rate_LBFGS, max_iter=max_iter_LBFGS)
    optimizer = torch.optim.Adam(dem.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(dem.parameters(), lr=learning_rate_SGD)

    # 定义学习率调度器
    lr_history = []
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)  # 每1000个epoch将学习率降低为原来的0.1倍
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)  # 指数衰减：每个epoch衰减为当前lr * gamma
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,  eta_min=1e-6 )  # 余弦退火：T_max为半周期（epoch数），eta_min为最小学习率
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.1, patience=10, threshold=1e-4)  # 按指标衰减：当loss在patience个epoch内下降小于阈值threshold时，将学习率降低为原来的factor倍

    print(f"train: Net={cfg.depth}x{cfg.input_size}-{cfg.hidden_size}-{cfg.output_size}({cfg.latent_dim}), lr={learning_rate}, scheduler={cfg.lr_scheduler}, loss_weight={cfg.loss_weight}")
    tqdm_epoch = tqdm(range(start_epoch, epoch_num), desc='epoches',colour='red', dynamic_ncols=True)
    for epoch in range(start_epoch, epoch_num):
        total_loss = 0
        total_eloss = 0
        total_bloss = 0
        optimizer.zero_grad()

        for i in range(cfg.data_num):
            # # 计算损失函数
            loss=Loss(dem)
            loss_value, energy_loss, boundary_loss=loss.loss_function(i, dom[i], bc_Dir[i], bc_Pre[i], bc_Sym[i])
            # 反向传播
            # loss_value.backward()
            total_loss += loss_value/cfg.data_num
            total_eloss += energy_loss/cfg.data_num
            total_bloss += boundary_loss/cfg.data_num

        total_loss.backward()  # 反向传播
        # 更新参数
        optimizer.step()
        scheduler.step()  # 更新学习率

            # def closure():
            #     loss=Loss(dem)
            #     loss_closure=loss.loss_function(dom, bc_Dir, bc_Pre, bc_Sym)
            #     # 反向传播
            #     optimizer.zero_grad()
            #     loss_closure.backward()
            #     return loss_closure
            
            # optimizer.step(closure=closure)

        losses.append(total_loss.item())
        lr_history.append(optimizer.param_groups[0]['lr'])


        # 更新epoch进度条
        tqdm_epoch.update()
        tqdm_epoch.set_postfix({'loss':'{:.5f}'.format(losses[-1]),'eloss':'{:.5f}'.format(total_eloss), 'bloss':'{:.5f}'.format(total_bloss), 'lr':'{:.5f}'.format(lr_history[-1])})

        if epoch % 2000 == 0:
            # 保存模型
            os.makedirs(cfg.model_save_path, exist_ok=True)
            torch.save(dem.state_dict(), f"{cfg.model_save_path}/dem_epoch{epoch}.pth")
            plot_loss(losses, lr_history)
            plt.savefig(f"{cfg.model_save_path}/training_curve_middle.png")
            with open(f"{cfg.model_save_path}/loss_middle_seed{cfg.seed}.txt", 'w') as f:
                f.write('\n'.join(map(str, losses)) + '\n')
    
    os.makedirs(cfg.model_save_path, exist_ok=True)
    torch.save(dem.state_dict(), f"{cfg.model_save_path}/dem_epoch{epoch_num}.pth")
    plt.savefig(f"{cfg.model_save_path}/training_curve_epoch{epoch_num}.png")
    with open(f"{cfg.model_save_path}/loss_epoch{epoch_num}_seed{cfg.seed}.txt", 'w') as f:
        f.write('\n'.join(map(str, losses)) + '\n')
    # with open(f"{cfg.model_save_path}/errorL2_epoch{epoch_num}_seed{cfg.seed}.txt", 'w') as f:
        # f.write('\n'.join(map(str, eL2)) + '\n')
    # with open(f"{cfg.model_save_path}/errorH1_epoch{epoch_num}_seed{cfg.seed}.txt", 'w') as f:
        # f.write('\n'.join(map(str, eH1)) + '\n')
    with open(f"{cfg.model_save_path}/lr_epoch{epoch}_seed{cfg.seed}.txt", 'w') as f:
        f.write('\n'.join(map(str, lr_history)) + '\n')

    end_time=time.time()-start_time
    print("End time: %.5f" % end_time)
    print("训练结束：结果保存在%s" % cfg.model_save_path)