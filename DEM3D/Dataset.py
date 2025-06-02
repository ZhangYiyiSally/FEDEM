import numpy as np
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self):
        self.dom=None
        pass

    # setup_domain: Create a 3D domain to be used as input for the neural network
    def setup_domain(self, lx: float, ly: float, lz: float, Nx: int, Ny: int, Nz: int):
        x_min, y_min, z_min = (0.0, 0.0, 0.0)
        x_dom = x_min, lx, Nx
        y_dom = y_min, ly, Ny
        z_dom = z_min, lz, Nz
        # create points
        x_space = np.linspace(x_dom[0], x_dom[1], x_dom[2])
        y_space = np.linspace(y_dom[0], y_dom[1], y_dom[2])
        z_space = np.linspace(z_dom[0], z_dom[1], z_dom[2])
        dom = np.zeros((Nx * Ny * Nz, 3))
        c = 0
        for z in np.nditer(z_space):
            for x in np.nditer(x_space):
                tb = y_dom[2] * c
                te = tb + y_dom[2]
                c += 1
                dom[tb:te, 0] = x
                dom[tb:te, 1] = y_space
                dom[tb:te, 2] = z
        self.dom=dom  
        return x_space, y_space, z_space, self.dom
    
    # bc_Dirichlet: Apply Dirichlet boundary conditions to the domain
    def bc_Dirichlet(self, marker, Dir_value:np.ndarray) -> dict:
        bc_idx = np.where(self.dom[:, 0] == marker)
        bc_coord = self.dom[bc_idx, :][0]
        bc_value = np.ones(np.shape(bc_coord)) * [Dir_value[0], Dir_value[1], Dir_value[2]]

        boundary_dirichlet = {
            # condition on the left
            "dirichlet_1": {
                "coord": bc_coord,
                "known_value": bc_value,
            }
        # adding more boundary condition here ...
        }
        return boundary_dirichlet
    
    # bc_Neumann: Apply Neumann boundary conditions to the domain
    def bc_Neumann(self, marker, Neu_value:np.ndarray) -> dict:
        bc_idx = np.where(self.dom[:, 0] == marker)
        bc_coord = self.dom[bc_idx, :][0]
        bc_value = np.ones(np.shape(bc_coord)) * [Neu_value[0], Neu_value[1], Neu_value[2]]

        boundary_neumann = {
        # condition on the right
            "neumann_1": {
                "coord": bc_coord,
                "known_value": bc_value,
            }
        # adding more boundary condition here ...
        }
        return boundary_neumann
    
    def datatest(self, lx_test, ly_test, lz_test, Nx_test: int, Ny_test: int, Nz_test: int):
        x_space, y_space, z_space, data=self.setup_domain(lx_test, ly_test, lz_test, Nx_test, Ny_test, Nz_test)
        return x_space, y_space, z_space, data
   
if __name__ == '__main__':
    dataset = Dataset()
    x, y, z, dom = dataset.setup_domain(lx=4.0, ly=1.0, lz=1.0, Nx=100, Ny=25, Nz=25)
    dirichlet=dataset.bc_Dirichlet(0.0, [0.0, 0.0, 0.0])
    neumann=dataset.bc_Neumann(4.0, [0.0, -5.0, 0.0])
    x_test, y_test, z_test, test=dataset.datatest(lx_test=4.0, ly_test=1.0, lz_test=1.0, Nx_test=3, Ny_test=3, Nz_test=3)
    
    # np.meshgrid(lin_x, lin_y, lin_z)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dom[:, 0], dom[:, 1], dom[:, 2], s=1, facecolor='blue')
    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.set_zlabel('Z', fontsize=3)
    ax.tick_params(labelsize=4)
    plt.show() 