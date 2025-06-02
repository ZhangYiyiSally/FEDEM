from torch.autograd import grad
import torch
import numpy as np
import torch.nn as nn
import Config as cfg


class Loss:
    def __init__(self, model):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model
        pass
        
    def loss_function(self, xyz_field: torch.Tensor, boundary_dirichlet: dict, boundary_neumann: dict, volume: list, bc_neu_area: list):
        internal_energy = self.StrainEnergy(xyz_field, volume)
        external_work = self.ExternalWork(boundary_neumann, bc_neu_area)
        boundary_loss = self.BoundaryLoss(boundary_dirichlet)

        energy_loss = internal_energy - external_work
        loss = energy_loss + 100*boundary_loss
        return loss

    def GetU(self, xyz_field: torch.Tensor):
        u = self.model(xyz_field)
        # Ux = xyz_field[:, 0] * u[:, 0]  # 强制边界条件
        # Uy = xyz_field[:, 0] * u[:, 1]
        # Uz = xyz_field[:, 0] * u[:, 2]
        Ux = u[:, 0]
        Uy = u[:, 1]
        Uz = u[:, 2]
        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        Uz = Uz.reshape(Uz.shape[0], 1)
        u_pred = torch.cat((Ux, Uy, Uz), -1)
        return u_pred
    
    def StrainEnergy(self, xyz_field: torch.Tensor, field_volume: list):
        E=cfg.E
        nu=cfg.nu
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        pred_u = self.GetU(xyz_field)

        duxdxyz = grad(pred_u[:, 0].unsqueeze(1), xyz_field, torch.ones(xyz_field.size()[0], 1, device=self.dev), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(pred_u[:, 1].unsqueeze(1), xyz_field, torch.ones(xyz_field.size()[0], 1, device=self.dev), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(pred_u[:, 2].unsqueeze(1), xyz_field, torch.ones(xyz_field.size()[0], 1, device=self.dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2

        strainenergy = 0.5 * lam * (torch.log(detF) * torch.log(detF)) - mu * torch.log(detF) + 0.5 * mu * (trC - 3)
        InternalEnergy = self.Intergration3D(strainenergy, field_volume[0], field_volume[1], field_volume[2])
    
        return InternalEnergy
    
    def ExternalWork(self, boundary_neumann: dict, bc_neu_area: list):
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        for i, keyi in enumerate(boundary_neumann):
            neuBC_coordinates[i] = torch.from_numpy(boundary_neumann[keyi]['coord']).float().to(self.dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(boundary_neumann[keyi]['known_value']).float().to(self.dev)

        external_W = torch.zeros(len(neuBC_coordinates))
        for i, vali in enumerate(neuBC_coordinates):
            pred_u_neu = self.GetU(neuBC_coordinates[i])
            fext = torch.bmm(pred_u_neu.unsqueeze(1), neuBC_values[i].unsqueeze(2))
            external_W[i] = self.montecarlo2D(fext, bc_neu_area[0], bc_neu_area[1])

        ExternalEnergy = torch.sum(external_W)
        return ExternalEnergy
    
    def BoundaryLoss(self, boundary_dirichlet: dict):
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        for i, keyi in enumerate(boundary_dirichlet):
            dirBC_coordinates[i] = torch.from_numpy(boundary_dirichlet[keyi]['coord']).float().to(self.dev)
            dirBC_values[i] = torch.from_numpy(boundary_dirichlet[keyi]['known_value']).float().to(self.dev)

        bc_u_loss = torch.zeros((len(dirBC_coordinates)))
        for i, vali in enumerate(dirBC_coordinates):
            pred_u_dir = self.GetU(dirBC_coordinates[i])
            mes_loss = nn.MSELoss(reduction='sum')
            bc_u_loss[i] = mes_loss(pred_u_dir, dirBC_values[i])

        boundary_loss = torch.sum(bc_u_loss)
        return boundary_loss
    
    def Intergration3D(self, strainenergy, lx, ly, lz):
        volume = lx * ly * lz
        return volume*torch.sum(strainenergy) / strainenergy.data.nelement()
    
    def montecarlo2D(self, fxy, lx, ly):
        area = lx * ly
        return area * torch.sum(fxy) / fxy.data.nelement()
    
if __name__ == '__main__':
    # 创建空间坐标xyz_field
    lx, ly, lz = 4.0, 1.0, 1.0 #立方体的长宽高
    nx_points, ny_points, nz_points = 100, 25, 25 #边上的节点数
    x_space = np.linspace(0,lx,nx_points) #x方向节点分布，向量
    y_space = np.linspace(0,ly,ny_points) #y方向节点分布，向量
    z_space = np.linspace(0,lz,nz_points) #z方向节点分布，向量
    dom = np.zeros((nx_points*ny_points*nz_points, 3))
    c = 0
    for z in np.nditer(z_space):
        for x in np.nditer(x_space):
            tb = ny_points * c
            te = tb + ny_points
            c += 1
            dom[tb:te, 0] = x
            dom[tb:te, 1] = y_space
            dom[tb:te, 2] = z 
    xyz_field=torch.tensor(dom, dtype=torch.float32) 

    # 创建速度场
    u=np.random.rand(nx_points*ny_points*nz_points, 3)
    u=torch.tensor(u, dtype=torch.float32)
    
    print('Done')