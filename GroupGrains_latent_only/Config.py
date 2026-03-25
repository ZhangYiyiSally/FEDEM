#---------------------------------seed-----------------------------------------
seed = 2025
#---------------------------------mesh-----------------------------------------
data_num=5
model_scale=10
#--------------------------------integration-----------------------------------
n_int3D=2
n_int2D=2

#-------------------------------material----------------------------------------
E=5e6
nu=0.498
#-------------------------------Dirichlet BC------------------------------------
Dir_marker='OutSurface'
Dir_u=[0.0, 0.0, 0.0]
#-------------------------------Pressure BC-------------------------------------
Pre_marker='InSurface'
Pre_value=[0.2e6, 7e6, 6.8e6]  # [initial, max, step], unit: Pa
Pre_step_interval=50000
#-------------------------------Symmetry BC-------------------------------------
Sym_marker='Symmetry'
#--------------------------------network----------------------------------------
input_size=3
hidden_size=200
output_size=3
depth=4
latent_dim=256
#--------------------------------training---------------------------------------
epoch_num=100000
lr=4e-4

# --------- pretrained and freezing (latent_vectors only) ---------
pretrained_model_path='GroupGrains_latent_only/models/dem_epoch100000.pth'
train_latent_only=False   # True: freeze all params except latent_vectors
load_pretrained_latent_vectors=False  # True: also load latent_vectors if shape matches

lr_scheduler='Exp'
gamma=0.9999
T_max=10000
eta_min=1e-6
loss_weight=[1e5, 1e8, 1e8-1e5]
weight_step_interval=50000

#---------------------------paths-----------------------------------------------
mesh_path=f"GroupGrains_latent_only/models"
if lr_scheduler == 'Cos':
    model_save_path=f"GroupGrains_latent_only/Results/DataNum{data_num}/x{model_scale}_Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr:.0e}_{T_max}x{eta_min:.1e}/p[{Pre_value[0]/1e6}-{Pre_value[2]/1e6}-{Pre_step_interval:.0f}]xw[{loss_weight[0]:.0e}-{loss_weight[2]:.0e}-{weight_step_interval:.0f}]"
    Evaluate_save_path=f"GroupGrains_latent_only/Results/DataNum{data_num}/x{model_scale}_Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr:.0e}_{T_max}x{eta_min:.1e}/p[{Pre_value[0]/1e6}-{Pre_value[2]/1e6}-{Pre_step_interval:.0f}]xw[{loss_weight[0]:.0e}-{loss_weight[2]:.0e}-{weight_step_interval:.0f}]"
if lr_scheduler == 'Exp':
    model_save_path=f"GroupGrains_latent_only/Results/DataNum{data_num}/x{model_scale}_Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr:.0e}_{gamma}/p[{Pre_value[0]/1e6}-{Pre_value[2]/1e6}-{Pre_step_interval:.0f}]xw[{loss_weight[0]:.0e}-{loss_weight[2]:.0e}-{weight_step_interval:.0f}]"
    Evaluate_save_path=f"GroupGrains_latent_only/Results/DataNum{data_num}/x{model_scale}_Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr:.0e}_{gamma}/p[{Pre_value[0]/1e6}-{Pre_value[2]/1e6}-{Pre_step_interval:.0f}]xw[{loss_weight[0]:.0e}-{loss_weight[2]:.0e}-{weight_step_interval:.0f}]"
