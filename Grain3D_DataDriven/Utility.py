import numpy as np
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTetra
import torch
from torch.autograd import grad
import meshio
from Network import ResNet
import Config as cfg

def FEMmeshtoVTK(filename, mesh, pointdata=None, celldata=None, fielddata=None):
    x=np.ascontiguousarray(mesh.points[:,0])
    y=np.ascontiguousarray(mesh.points[:,1])
    z=np.ascontiguousarray(mesh.points[:,2])
    # Define connectivity or vertices that belongs to each element
    conn=mesh.cells_dict['tetra'].flatten()
    # Define offset of last vertex of each element
    cell_nodes=np.full(mesh.cells_dict['tetra'].shape[0], mesh.cells_dict['tetra'].shape[1])
    offset=np.cumsum(cell_nodes)
    # Define cell types
    ctype = np.full(len(mesh.cells_dict['tetra']), VtkTetra.tid)

    # Write to file
    unstructuredGridToVTK(
        filename,
        x,
        y,
        z,
        connectivity=conn,
        offsets=offset,
        cell_types=ctype,
        cellData=celldata,
        pointData=pointdata,
        fieldData=fielddata,
    )

def errorL2(model,xyz,dev):
    Uref_L2=4.428023963208778E-3

    xyz_tensor = torch.from_numpy(xyz).float()
    xyz_tensor = xyz_tensor.to(dev)
    U=model(xyz_tensor)
    U_L2norm = torch.mean(torch.sqrt(U[:, 0]**2 + U[:, 1]**2 + U[:, 2]**2))
    eL2=abs(U_L2norm-Uref_L2)/Uref_L2
    return eL2

def errorH1(model,xyz,dev):
    Uref_H1=1.732141582336874E0

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

    U_H1norm = torch.mean(torch.sqrt(F11**2 + F12**2 + F13**2 + F21**2 + F22**2 + F23**2 + F31**2 + F32**2 + F33**2))
    eH1=abs(U_H1norm-Uref_H1)/Uref_H1
    return eH1

def read_data_file(file_path, num):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 跳过以%开头的注释行
            if not line.strip().startswith('%'):
                # 处理数据行，按空格分割
                values = [float(val) for val in line.strip().split()]
                data.append(values)
    if data:
        data = np.array(data)
        # 前3列为坐标，后3列为位移
        reshape_data = data.reshape(6, num).T
        return reshape_data
    else:
        # 如果没有数据，返回两个空数组
        return np.array([]), np.array([])

