import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.autograd import grad
import meshio
import numpy as np
import Config as cfg
from Dataset import Dataset
from Network import ResNet
from Loss import Loss
import Utility as util

# 选择GPU或CPU
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate_model(model, xyz):
    xyz_tensor = torch.from_numpy(xyz).float()
    xyz_tensor = xyz_tensor.to(dev)
    xyz_tensor.requires_grad_(True)

    loss=Loss(model)
    # 计算位移
    u_pred_torch = loss.GetU(xyz_tensor)
    u_pred = u_pred_torch.detach().cpu().numpy()
    U = (np.float64(u_pred[:,0]), np.float64(u_pred[:,1]), np.float64(u_pred[:,2]))

    #计算应力
    E=cfg.E
    nu =cfg.nu
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    duxdxyz = grad(u_pred_torch[:, 0].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duydxyz = grad(u_pred_torch[:, 1].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duzdxyz = grad(u_pred_torch[:, 2].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    # 计算变形梯度F
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
    
    # 计算第一类皮奥拉-基尔霍夫应力P
    P11 = mu * F11 + (lam * torch.log(detF) - mu) * invF11
    P12 = mu * F12 + (lam * torch.log(detF) - mu) * invF21
    P13 = mu * F13 + (lam * torch.log(detF) - mu) * invF31
    P21 = mu * F21 + (lam * torch.log(detF) - mu) * invF12
    P22 = mu * F22 + (lam * torch.log(detF) - mu) * invF22
    P23 = mu * F23 + (lam * torch.log(detF) - mu) * invF32
    P31 = mu * F31 + (lam * torch.log(detF) - mu) * invF13
    P32 = mu * F32 + (lam * torch.log(detF) - mu) * invF23
    P33 = mu * F33 + (lam * torch.log(detF) - mu) * invF33

    # 计算第二类皮奥拉-基尔霍夫应力S
    S11 = invF11 * P11 + invF12 * P21 + invF13 * P31
    S12 = invF11 * P12 + invF12 * P22 + invF13 * P32
    S13 = invF11 * P13 + invF12 * P23 + invF13 * P33
    S22 = invF21 * P12 + invF22 * P22 + invF23 * P32
    S23 = invF21 * P13 + invF22 * P23 + invF23 * P33
    S33 = invF31 * P13 + invF32 * P23 + invF33 * P33

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
    E22_pred = E22.detach().cpu().numpy()
    E23_pred = E23.detach().cpu().numpy()
    E33_pred = E33.detach().cpu().numpy()
    S11_pred = S11.detach().cpu().numpy()
    S12_pred = S12.detach().cpu().numpy()
    S13_pred = S13.detach().cpu().numpy()
    S22_pred = S22.detach().cpu().numpy()
    S23_pred = S23.detach().cpu().numpy()
    S33_pred = S33.detach().cpu().numpy()
    SVonMises_tmp = np.float64(np.sqrt(0.5 * ((S11_pred - S22_pred) ** 2 + (S22_pred - S33_pred) ** 2 + (S33_pred - S11_pred) ** 2 + 6 * (S12_pred ** 2 + S23_pred ** 2 + S13_pred ** 2))))
    SVonMises=SVonMises_tmp[:,0]
    return U, SVonMises, \
        S11_pred[:,0], S12_pred[:,0], S13_pred[:,0], \
        S23_pred[:,0], S22_pred[:,0], S33_pred[:,0], \
        E11_pred[:,0], E12_pred[:,0], E13_pred[:,0], \
        E22_pred[:,0], E23_pred[:,0], E33_pred[:,0]

# 从文件加载已经训练完成的模型
# model=MultiLayerNet(D_in=3, H=30, D_out=3).cuda()
model=ResNet(input_size=3, hidden_size=64, output_size=3, depth=4).cuda()
model.load_state_dict(torch.load(f"{cfg.model_save_path}/dem.pth"))
model.eval()  # 设置模型为evaluation状态

# 读取有限元网格
# mesh0=meshio.read("DEFEM3D/Beam3D/beam_mesh0.msh", file_format="gmsh")
mesh = meshio.read(cfg.mesh_path, file_format="gmsh")

# 计算该有限元网格对应的预测值
xyz=mesh.points
U, SVonMises, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33= evaluate_model(model, xyz)

# all_cells=mesh.cell_data['gmsh:physical'][6]
# 写入vtu网格文件
util.FEMmeshtoVTK(cfg.Evaluate_save_path, mesh, pointdata={"U": U, "SVonMises": SVonMises, \
                                                           "S11": S11, "S12": S12, "S13": S13, \
                                                           "S22": S22, "S23": S23, "S33": S33, \
                                                           "E11": E11, "E12": E12, "E13": E13, \
                                                           "E22": E22, "E23": E23, "E33": E33}, \
                                                )

U_data=np.hstack((xyz, np.linalg.norm(np.array(U).T, axis=1).reshape(-1,1)))
SVonMises_data=np.hstack((xyz, SVonMises.reshape(-1,1)))

np.savetxt(cfg.model_save_path+'/U.txt', U_data, fmt='%f', delimiter=' ')
np.savetxt(cfg.model_save_path+'/SVonMises.txt', SVonMises_data, fmt='%f', delimiter=' ')
print('结果已经保存在'+cfg.model_save_path)
