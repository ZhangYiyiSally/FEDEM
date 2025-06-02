import numpy as np
import torch
import meshio
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def domain(self, mesh) -> torch.Tensor:
        self.mesh=mesh
        AllPoint_idx=mesh.cells_dict['tetra']
        Tetra_coord=mesh.points[AllPoint_idx]
        Tetra_coord=torch.tensor(Tetra_coord, dtype=torch.float32).to(self.dev)
        return Tetra_coord
        
    def bc_Dirichlet(self, marker:str) -> torch.Tensor:
        DirCell_idx=self.mesh.cell_sets_dict[marker]['triangle']
        DirPoint_idx=self.mesh.cells_dict['triangle'][DirCell_idx]
        Dir_Triangle_coord = self.mesh.points[DirPoint_idx]
        Dir_Triangle_coord=torch.tensor(Dir_Triangle_coord, dtype=torch.float32).to(self.dev)
        return Dir_Triangle_coord
    
    def bc_Neumann(self, marker:str) -> torch.Tensor:
        NeuCell_idx=self.mesh.cell_sets_dict[marker]['triangle']
        NeuPoint_idx=self.mesh.cells_dict['triangle'][NeuCell_idx]
        Neu_Triangle_coord = self.mesh.points[NeuPoint_idx]
        Neu_Triangle_coord=torch.tensor(Neu_Triangle_coord, dtype=torch.float32).to(self.dev)
        return Neu_Triangle_coord
    
if __name__ == '__main__':  # 测试边界条件是否设置正确
    mesh = meshio.read("DEFEM3D/Beam3D/beam_mesh.msh", file_format="gmsh")
    data = Dataset()
    dom = data.domain(mesh)
    Dir_coord = data.bc_Dirichlet('bc_Dirichlet')
    Neu_coord = data.bc_Neumann('bc_Neumann')

    print("全域四面体单元个数*单元顶点个数*坐标方向:", dom.shape)
    print("Dirichlet边界三角形单元个数*单元顶点个数*坐标方向:", Dir_coord.shape)
    print("Neumann边界三角形单元个数*单元顶点个数*坐标方向:", Neu_coord.shape)
    
    