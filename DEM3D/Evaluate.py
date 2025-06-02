import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.autograd import grad
import Config as cfg
from Dataset import Dataset
from Network import ResNet
import Utility
from pyevtk.hl import gridToVTK
from Loss import Loss

# 选择GPU或CPU
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate_model(model, x, y, z):
    E = cfg.E
    nu = cfg.nu
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    Nx = len(x)
    Ny = len(y)
    Nz = len(z)
    xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
    x1D = xGrid.flatten()
    y1D = yGrid.flatten()
    z1D = zGrid.flatten()
    xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
    xyz_tensor = torch.from_numpy(xyz).float()
    xyz_tensor = xyz_tensor.to(dev)
    xyz_tensor.requires_grad_(True)

    loss=Loss(model)
    u_pred_torch = loss.GetU(xyz_tensor)
    duxdxyz = grad(u_pred_torch[:, 0].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duydxyz = grad(u_pred_torch[:, 1].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duzdxyz = grad(u_pred_torch[:, 2].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    F11 = duxdxyz[:, 0].unsqueeze(1) + 1
    F12 = duxdxyz[:, 1].unsqueeze(1) + 0
    F13 = duxdxyz[:, 2].unsqueeze(1) + 0
    F21 = duydxyz[:, 0].unsqueeze(1) + 0
    F22 = duydxyz[:, 1].unsqueeze(1) + 1
    F23 = duydxyz[:, 2].unsqueeze(1) + 0
    F31 = duzdxyz[:, 0].unsqueeze(1) + 0
    F32 = duzdxyz[:, 1].unsqueeze(1) + 0
    F33 = duzdxyz[:, 2].unsqueeze(1) + 1
    detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
    invF11 = (F22 * F33 - F23 * F32) / detF
    invF12 = -(F12 * F33 - F13 * F32) / detF
    invF13 = (F12 * F23 - F13 * F22) / detF
    invF21 = -(F21 * F33 - F23 * F31) / detF
    invF22 = (F11 * F33 - F13 * F31) / detF
    invF23 = -(F11 * F23 - F13 * F21) / detF
    invF31 = (F21 * F32 - F22 * F31) / detF
    invF32 = -(F11 * F32 - F12 * F31) / detF
    invF33 = (F11 * F22 - F12 * F21) / detF
    C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
    C12 = F11 * F12 + F21 * F22 + F31 * F32
    C13 = F11 * F13 + F21 * F23 + F31 * F33
    C21 = F12 * F11 + F22 * F21 + F32 * F31
    C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
    C23 = F12 * F13 + F22 * F23 + F32 * F33
    C31 = F13 * F11 + F23 * F21 + F33 * F31
    C32 = F13 * F12 + F23 * F22 + F33 * F32
    C33 = F13 ** 2 + F23 ** 2 + F33 ** 2
    E11 = 0.5 * (C11 - 1)
    E12 = 0.5 * C12
    E13 = 0.5 * C13
    E21 = 0.5 * C21
    E22 = 0.5 * (C22 - 1)
    E23 = 0.5 * C23
    E31 = 0.5 * C31
    E32 = 0.5 * C32
    E33 = 0.5 * (C33 - 1)
    
    P11 = mu * F11 + (lam * torch.log(detF) - mu) * invF11
    P12 = mu * F12 + (lam * torch.log(detF) - mu) * invF21
    P13 = mu * F13 + (lam * torch.log(detF) - mu) * invF31
    P21 = mu * F21 + (lam * torch.log(detF) - mu) * invF12
    P22 = mu * F22 + (lam * torch.log(detF) - mu) * invF22
    P23 = mu * F23 + (lam * torch.log(detF) - mu) * invF32
    P31 = mu * F31 + (lam * torch.log(detF) - mu) * invF13
    P32 = mu * F32 + (lam * torch.log(detF) - mu) * invF23
    P33 = mu * F33 + (lam * torch.log(detF) - mu) * invF33

    S11 = invF11 * P11 + invF12 * P21 + invF13 * P31
    S12 = invF11 * P12 + invF12 * P22 + invF13 * P32
    S13 = invF11 * P13 + invF12 * P23 + invF13 * P33
    S21 = invF21 * P11 + invF22 * P21 + invF23 * P31
    S22 = invF21 * P12 + invF22 * P22 + invF23 * P32
    S23 = invF21 * P13 + invF22 * P23 + invF23 * P33
    S31 = invF31 * P11 + invF32 * P21 + invF33 * P31
    S32 = invF31 * P12 + invF32 * P22 + invF33 * P32
    S33 = invF31 * P13 + invF32 * P23 + invF33 * P33

    u_pred = u_pred_torch.detach().cpu().numpy()
    F11_pred = F11.detach().cpu().numpy()
    F12_pred = F12.detach().cpu().numpy()
    F13_pred = F13.detach().cpu().numpy()
    F21_pred = F21.detach().cpu().numpy()
    F22_pred = F22.detach().cpu().numpy()
    F23_pred = F23.detach().cpu().numpy()
    F31_pred = F31.detach().cpu().numpy()
    F32_pred = F32.detach().cpu().numpy()
    F33_pred = F33.detach().cpu().numpy()
    E11_pred = E11.detach().cpu().numpy()
    E12_pred = E12.detach().cpu().numpy()
    E13_pred = E13.detach().cpu().numpy()
    E21_pred = E21.detach().cpu().numpy()
    E22_pred = E22.detach().cpu().numpy()
    E23_pred = E23.detach().cpu().numpy()
    E31_pred = E31.detach().cpu().numpy()
    E32_pred = E32.detach().cpu().numpy()
    E33_pred = E33.detach().cpu().numpy()
    S11_pred = S11.detach().cpu().numpy()
    S12_pred = S12.detach().cpu().numpy()
    S13_pred = S13.detach().cpu().numpy()
    S21_pred = S21.detach().cpu().numpy()
    S22_pred = S22.detach().cpu().numpy()
    S23_pred = S23.detach().cpu().numpy()
    S31_pred = S31.detach().cpu().numpy()
    S32_pred = S32.detach().cpu().numpy()
    S33_pred = S33.detach().cpu().numpy()
    surUx = u_pred[:, 0].reshape(Ny, Nx, Nz)
    surUy = u_pred[:, 1].reshape(Ny, Nx, Nz)
    surUz = u_pred[:, 2].reshape(Ny, Nx, Nz)
    surE11 = E11_pred.reshape(Ny, Nx, Nz)
    surE12 = E12_pred.reshape(Ny, Nx, Nz)
    surE13 = E13_pred.reshape(Ny, Nx, Nz)
    surE21 = E21_pred.reshape(Ny, Nx, Nz)
    surE22 = E22_pred.reshape(Ny, Nx, Nz)
    surE23 = E23_pred.reshape(Ny, Nx, Nz)
    surE31 = E31_pred.reshape(Ny, Nx, Nz)
    surE32 = E32_pred.reshape(Ny, Nx, Nz)
    surE33 = E33_pred.reshape(Ny, Nx, Nz)
    surS11 = S11_pred.reshape(Ny, Nx, Nz)
    surS12 = S12_pred.reshape(Ny, Nx, Nz)
    surS13 = S13_pred.reshape(Ny, Nx, Nz)
    surS21 = S21_pred.reshape(Ny, Nx, Nz)
    surS22 = S22_pred.reshape(Ny, Nx, Nz)
    surS23 = S23_pred.reshape(Ny, Nx, Nz)
    surS31 = S31_pred.reshape(Ny, Nx, Nz)
    surS32 = S32_pred.reshape(Ny, Nx, Nz)
    surS33 = S33_pred.reshape(Ny, Nx, Nz)
    SVonMises = np.float64(np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22 - surS33) ** 2 + (surS33 - surS11) ** 2 + 6 * (surS12 ** 2 + surS23 ** 2 + surS31 ** 2))))
    U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
    S1 = (np.float64(surS11), np.float64(surS12), np.float64(surS13))
    S2 = (np.float64(surS21), np.float64(surS22), np.float64(surS23))
    S3 = (np.float64(surS31), np.float64(surS32), np.float64(surS33))
    E1 = (np.float64(surE11), np.float64(surE12), np.float64(surE13))
    E2 = (np.float64(surE21), np.float64(surE22), np.float64(surE23))
    E3 = (np.float64(surE31), np.float64(surE32), np.float64(surE33))
    return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(surS23), \
           np.float64(surS33), np.float64(surE11), np.float64(surE12), \
           np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(SVonMises), \
           np.float64(F11_pred), np.float64(F12_pred), np.float64(F13_pred), \
           np.float64(F21_pred), np.float64(F22_pred), np.float64(F23_pred), \
           np.float64(F31_pred), np.float64(F32_pred), np.float64(F33_pred)

# 从文件加载已经训练完成的模型
# model=MultiLayerNet(D_in=3, H=30, D_out=3).cuda()
model=ResNet(input_size=3, hidden_size=64, output_size=3, depth=4).cuda()
model.load_state_dict(torch.load(f"{cfg.model_save_path}/dem.pth"))
model.eval()  # 设置模型为evaluation状态

# model_loaded = torch.load("DEM3D/Results/dem.pth", map_location=device)
# model_loaded.eval()  # 设置模型为evaluation状态

# 生成时空网格
nx_test=100
ny_test=25
nz_test=25
data=Dataset()
x, y, z, xyz=data.datatest(4.0, 1.0, 1.0, nx_test, ny_test, nz_test)
xyz_field=torch.tensor(xyz, dtype=torch.float32).cuda()

# 计算该时空网格对应的预测值
U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F13, F21, F22, F23, F31, F32, F33 = evaluate_model(model, x, y, z)

Utility.write_vtk_v2(cfg.Evaluate_save_path, x, y, z, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises)

Ureshape = np.array(U).reshape(3, xyz.shape[0])
U_data=np.hstack((xyz, np.linalg.norm(Ureshape.T, axis=1).reshape(-1,1)))
SVonMises_data=np.hstack((xyz, SVonMises.transpose(2, 1, 0).reshape(-1,1)))

np.savetxt(cfg.model_save_path+'/U.txt', U_data, fmt='%f', delimiter=' ')
np.savetxt(cfg.model_save_path+'/SVonMises.txt', SVonMises_data, fmt='%f', delimiter=' ')
print('结果已经保存在'+cfg.model_save_path)
