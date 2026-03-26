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
        # Ux = xyz_field[:, 0] * u[:, 0]  # Enforce boundary condition
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
        method = getattr(cfg, "integration_method", "mean").lower()
        lx, ly, lz = field_volume[0], field_volume[1], field_volume[2]

        if method in ("mean", "montecarlo", "mc"):
            InternalEnergy = self.Intergration3D(strainenergy, lx, ly, lz)
        elif method in ("trapezoidal", "trapz", "trapezoid"):
            InternalEnergy = self.Intergration3D_trapezoidal(strainenergy, lx, ly, lz, cfg.Nx, cfg.Ny, cfg.Nz)
        elif method in ("simpson", "simpson-type", "simpson_type"):
            InternalEnergy = self.Intergration3D_simpson(strainenergy, lx, ly, lz, cfg.Nx, cfg.Ny, cfg.Nz)
        else:
            raise ValueError(
                f"Unknown integration_method: {method}. Use 'mean', 'trapezoidal', or 'simpson'."
            )

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

    def _reshape_integrand_3d(self, integrand, nx, ny, nz):
        total_points = int(nx) * int(ny) * int(nz)
        if integrand.numel() != total_points:
            raise ValueError(
                f"Integrand size ({integrand.numel()}) does not match nx*ny*nz ({total_points})."
            )
        # The mesh is flattened with y-fastest, then x, then z.
        return integrand.reshape(int(nz), int(nx), int(ny))

    def Intergration3D_trapezoidal(self, integrand, lx, ly, lz, nx=None, ny=None, nz=None):
        nx = cfg.Nx if nx is None else nx
        ny = cfg.Ny if ny is None else ny
        nz = cfg.Nz if nz is None else nz

        if nx < 2 or ny < 2 or nz < 2:
            raise ValueError("Trapezoidal integration requires nx, ny, nz >= 2.")

        grid = self._reshape_integrand_3d(integrand, nx, ny, nz)
        dx = lx / (nx - 1)
        dy = ly / (ny - 1)
        dz = lz / (nz - 1)

        int_y = torch.trapz(grid, dx=dy, dim=2)
        int_xy = torch.trapz(int_y, dx=dx, dim=1)
        int_xyz = torch.trapz(int_xy, dx=dz, dim=0)
        return int_xyz

    def Intergration3D_simpson(self, integrand, lx, ly, lz, nx=None, ny=None, nz=None):
        nx = cfg.Nx if nx is None else nx
        ny = cfg.Ny if ny is None else ny
        nz = cfg.Nz if nz is None else nz

        if nx < 3 or ny < 3 or nz < 3:
            raise ValueError("Simpson integration requires nx, ny, nz >= 3.")
        if (nx - 1) % 2 != 0 or (ny - 1) % 2 != 0 or (nz - 1) % 2 != 0:
            raise ValueError("Simpson integration requires even interval counts in each direction.")

        grid = self._reshape_integrand_3d(integrand, nx, ny, nz)
        dx = lx / (nx - 1)
        dy = ly / (ny - 1)
        dz = lz / (nz - 1)

        wx = torch.ones(nx, dtype=grid.dtype, device=grid.device)
        wy = torch.ones(ny, dtype=grid.dtype, device=grid.device)
        wz = torch.ones(nz, dtype=grid.dtype, device=grid.device)

        wx[1:-1:2] = 4.0
        wx[2:-1:2] = 2.0
        wy[1:-1:2] = 4.0
        wy[2:-1:2] = 2.0
        wz[1:-1:2] = 4.0
        wz[2:-1:2] = 2.0

        weight_3d = wz[:, None, None] * wx[None, :, None] * wy[None, None, :]
        weighted_sum = torch.sum(grid * weight_3d)
        return weighted_sum * dx * dy * dz / 27.0    
    def montecarlo2D(self, fxy, lx, ly):
        area = lx * ly
        return area * torch.sum(fxy) / fxy.data.nelement()
    
if __name__ == '__main__':
    # Build spatial coordinates xyz_field
    lx, ly, lz = 4.0, 1.0, 1.0 # Cuboid dimensions
    nx_points, ny_points, nz_points = 100, 25, 25 # Number of points on each axis
    x_space = np.linspace(0,lx,nx_points) # x-axis point distribution
    y_space = np.linspace(0,ly,ny_points) # y-axis point distribution
    z_space = np.linspace(0,lz,nz_points) # z-axis point distribution
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

    # Build a random displacement field
    u=np.random.rand(nx_points*ny_points*nz_points, 3)
    u=torch.tensor(u, dtype=torch.float32)
    
    print('Done')



