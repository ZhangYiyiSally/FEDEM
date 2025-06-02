import torch
from torch.autograd import grad
import numpy as np
from pyevtk.hl import gridToVTK
import meshio
from Network import ResNet
import Config as cfg
from Dataset import Dataset

# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_vtk(filename, x_space, y_space, z_space, U):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})

# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_vtk_v2(filename, x_space, y_space, z_space, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U, "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33, \
                                               "E11": E11, "E12": E12, "E13": E13, \
                                               "E22": E22, "E23": E23, "E33": E33\
                                               })


def errorL2(model,xyz,dev):
    Uref_L2=4.786440977217170E-1

    xyz_tensor = torch.from_numpy(xyz).float()
    xyz_tensor = xyz_tensor.to(dev)
    U=model(xyz_tensor)
    U_L2norm = torch.mean(torch.sqrt(U[:, 0]**2 + U[:, 1]**2 + U[:, 2]**2))
    eL2=abs(U_L2norm-Uref_L2)/Uref_L2
    return eL2

def errorH1(model,xyz,dev):
    Uref_H1=1.733104848929538E0

    xyz_tensor = torch.from_numpy(xyz).float()
    xyz_tensor = xyz_tensor.to(dev)
    xyz_tensor.requires_grad_(True)
    u_pred_torch=model(xyz_tensor)
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

    # Uxx=duxdxyz[:, 0]
    # Uxy=duxdxyz[:, 1]
    # Uxz=duxdxyz[:, 2]
    # Uyx=duydxyz[:, 0]
    # Uyy=duydxyz[:, 1]
    # Uyz=duydxyz[:, 2]
    # Uzx=duzdxyz[:, 0]
    # Uzy=duzdxyz[:, 1]
    # Uzz=duzdxyz[:, 2]

    U_H1norm = torch.mean(torch.sqrt(F11**2 + F12**2 + F13**2 + F21**2 + F22**2 + F23**2 + F31**2 + F32**2 + F33**2))
    # U_H1norm=torch.mean(torch.sqrt(Uxx**2 + Uxy**2 + Uxz**2 + Uyx**2 + Uyy**2 + Uyz**2 + Uzx**2 + Uzy**2 + Uzz**2))
    eH1=abs(U_H1norm-Uref_H1)/Uref_H1
    return eH1

if __name__ == '__main__':
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model=ResNet(input_size=3, hidden_size=64, output_size=3, depth=4).to(dev)
    model.load_state_dict(torch.load(f"{cfg.model_save_path}/dem.pth"))
    model.eval()

    # mesh = meshio.read("DEFEM3D/Beam3D/beam_mesh.msh", file_format="gmsh")
    # xyz = mesh.points

    data=Dataset()
    x_space, y_space, z_space, xyz=data.setup_domain(lx=cfg.Length, ly=cfg.Width, lz=cfg.Height, Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz)

    eL2=errorL2(model,xyz,dev)
    eH1=errorH1(model,xyz,dev)
    print(f"eL2={eL2}, eH1={eH1}")