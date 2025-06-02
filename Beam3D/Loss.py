from torch.autograd import grad
import torch
import numpy as np
import torch.nn as nn
import meshio
import time
from scipy.interpolate import griddata
import Config as cfg
from Network import ResNet
from Dataset import Dataset
from GaussIntegral import GaussIntegral


class Loss:
    def __init__(self, model):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model
        pass
        
    def loss_function(self, Tetra_coord: torch.Tensor, Dir_Triangle_coord: torch.Tensor, Neu_Triangle_coord: torch.Tensor) -> torch.Tensor:
        integral=GaussIntegral()
        integral_strainenergy=integral.Integral3D(self.StrainEnergy, cfg.n_int3D, Tetra_coord)
        integral_externalwork=integral.Integral2D(self.ExternalWork, cfg.n_int2D, Neu_Triangle_coord)
        # integral_boundaryloss=integral.Integral2D(self.BoundaryLoss, 3, Dir_Triangle_coord)
        integral_boundaryloss=self.BoundaryLoss(Dir_Triangle_coord)

        energy_loss = integral_strainenergy - integral_externalwork
        loss = energy_loss + 10*integral_boundaryloss
        
        # print("Internal Energy:", integral_strainenergy.item())
        # print("External Work:", integral_externalwork.item())
        # print("Boundary Loss:", integral_boundaryloss.item())
        return loss

    def GetU(self, xyz_field: torch.Tensor) -> torch.Tensor:
        u = self.model(xyz_field)
        return u
    
    def StrainEnergy(self, xyz_field: torch.Tensor) -> torch.Tensor:
        E=cfg.E
        nu=cfg.nu
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        xyz_field.requires_grad = True  # 为了计算位移场的梯度，这里需要设置为True
        pred_u = self.GetU(xyz_field)

        duxdxyz = grad(pred_u[:, :, 0], xyz_field, torch.ones_like(pred_u[:, :, 0]), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(pred_u[:, :, 1], xyz_field, torch.ones_like(pred_u[:, :, 1]), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(pred_u[:, :, 2], xyz_field, torch.ones_like(pred_u[:, :, 2]), create_graph=True, retain_graph=True)[0]
        # dudxyz=torch.cat((duxdxyz.unsqueeze(2), duydxyz.unsqueeze(2), duzdxyz.unsqueeze(2)), dim=2)
        # Identity=torch.eye(3, device=self.dev).view(1, 1, 3, 3)
        # F=dudxyz+Identity
        # detF=torch.det(F)
        Fxx = duxdxyz[:, :, 0].unsqueeze(2) + 1
        Fxy = duxdxyz[:, :, 1].unsqueeze(2) + 0
        Fxz = duxdxyz[:, :, 2].unsqueeze(2) + 0
        Fyx = duydxyz[:, :, 0].unsqueeze(2) + 0
        Fyy = duydxyz[:, :, 1].unsqueeze(2) + 1
        Fyz = duydxyz[:, :, 2].unsqueeze(2) + 0
        Fzx = duzdxyz[:, :, 0].unsqueeze(2) + 0
        Fzy = duzdxyz[:, :, 1].unsqueeze(2) + 0
        Fzz = duzdxyz[:, :, 2].unsqueeze(2) + 1
        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2

        strainenergy_tmp = 0.5 * lam * (torch.log(detF) * torch.log(detF)) - mu * torch.log(detF) + 0.5 * mu * (trC - 3)
        strainenergy = strainenergy_tmp[:, :, 0]
    
        return strainenergy
    
    def ExternalWork(self, neumann_field: torch.Tensor) -> torch.Tensor:
        #"""计算外力做功"""
        neumann_field.requires_grad = True
        u_pred = self.GetU(neumann_field)

        t_value=torch.tensor(cfg.Neu_t, dtype=torch.float32).to(self.dev)
        t=torch.zeros_like(u_pred)
        t[:, :, :]=t_value
        
        external_work = torch.sum(u_pred * t, dim=2)
        return external_work
    
    def BoundaryLoss(self, dirichlet_field: torch.Tensor) -> torch.Tensor:
        #"""计算边界条件损失函数"""
        u_pred = self.GetU(dirichlet_field)

        u_value=torch.tensor(cfg.Dir_u, dtype=torch.float32).to(self.dev)
        u_true = torch.zeros_like(u_pred)
        u_true[:, :, :]=u_value

        mes_loss = nn.MSELoss(reduction='sum')
        boundary_loss = mes_loss(u_pred, u_true)
        # boundary_loss = torch.sum((u_pred - u_true) ** 2, dim=2)
        return boundary_loss

if __name__ =='__main__':
    start_time=time.time()
    torch.manual_seed(2025)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dem=ResNet(input_size=3, hidden_size=64, output_size=3, depth=4).to(dev)

    mesh = meshio.read("DEFEM3D/Beam3D/beam_mesh.msh", file_format="gmsh")

    # 提取四面体单元顶点坐标数组：m*4*3（m为四面体单元个数，4为四面体的四个顶点，3为三维空间坐标）
    AllPoint_idx=mesh.cells_dict['tetra']
    Tetra_coord=mesh.points[AllPoint_idx]
    Tetra_coord=torch.tensor(Tetra_coord, dtype=torch.float32).to(dev)

    # 提取狄利克雷边界条件Dirichlet
    DirCell_idx=mesh.cell_sets_dict['bc_Dirichlet']['triangle']
    DirPoint_idx=mesh.cells_dict['triangle'][DirCell_idx]
    Dir_coord = mesh.points[DirPoint_idx]
    Dir_coord=torch.tensor(Dir_coord, dtype=torch.float32).to(dev)

    # 提取纽曼边界条件
    NeuCell_idx=mesh.cell_sets_dict['bc_Neumann']['triangle']
    NeuPoint_idx=mesh.cells_dict['triangle'][NeuCell_idx]
    Neu_coord = mesh.points[NeuPoint_idx]
    Neu_coord=torch.tensor(Neu_coord, dtype=torch.float32).to(dev)

    loss=Loss(dem)
    loss_value=loss.loss_function(Tetra_coord, Dir_coord, Neu_coord)
    end_time=time.time()
    print('损失函数值为：', loss_value.item())
    print("计算时间:", end_time-start_time, "s")