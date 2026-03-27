#---------------------------------Random seed---------------------------------
seed=2025

#--------------------------------Geometry settings-----------------------------
#--------Length (x direction), Width (y direction), Height (z direction)------
Length=4.0
Width=1.0
Height=1.0

#--------Number of points in each direction: Nx, Ny, Nz-----------------------
Nx=40
Ny=10
Nz=10

#--------Integration method: mean | trapezoidal | simpson---------------------
integration_method="mean"
# integration_method="trapezoidal"
# integration_method="simpson"

#-------------------------------Material parameters----------------------------
#--------E: Young's modulus, nu: Poisson's ratio------------------------------
E=1000
nu=0.3

#-------------------------------Dirichlet boundary condition-------------------
#--------Dir_marker: boundary marker, Dir_u: prescribed displacement----------
Dir_marker=0.0
Dir_u=[0.0, 0.0, 0.0]

#-------------------------------Neumann boundary condition---------------------
#--------Neu_marker: boundary marker, Neu_t: prescribed traction--------------
Neu_marker=4.0
Neu_t=[0.0, -5.0, 0.0]

#--------------------------------Neural network settings-----------------------
input_size=3  # ResNet input size
hidden_size=64  # ResNet hidden layer size
output_size=3  # ResNet output size
depth=4  # ResNet depth

#--------------------------------Training settings-----------------------------
epoch_num=2050  # Number of training epochs
beta=0.975 # Switch parameter between AdamW and LBFGS
learning_rate_Adam=0.0005  # Learning rate for Adam
gamma=0.99
learning_rate_LBFGS=0.2  # Learning rate for LBFGS
max_iter_LBFGS=20  # Maximum iterations for LBFGS

#---------------------------Output save paths----------------------------------
model_save_path=f"DEM3D/Results/{integration_method}/mesh{Nx}x{Ny}x{Nz}"
Evaluate_save_path=f"DEM3D/Results/{integration_method}/mesh{Nx}x{Ny}x{Nz}/NeoHook_{depth}Layer_mesh{Nx*Ny*Nz}_iter{epoch_num}_lr{learning_rate_Adam}_mont"
