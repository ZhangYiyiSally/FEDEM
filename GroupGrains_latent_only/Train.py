import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import time
import meshio
import Config as cfg
from Network import ResNet
from Dataset import Dataset
from Loss import Loss


def plot_loss(loss, error):
    plt.plot(loss, linewidth=2, color='firebrick')
    plt.plot(error, linewidth=2, color='blue')
    plt.tick_params(axis='both', size=5, width=2, direction='in', labelsize=15)
    plt.xlabel('epoch', size=15)
    plt.ylabel('Loss', size=15)
    plt.legend(['Loss', 'lr'], loc='upper right', fontsize=15)
    plt.title('Training Curve', size=20)
    plt.grid(color='midnightblue', linestyle='-.', linewidth=0.5)
    plt.ylim(-2000, 1)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.pause(0.0001)


def get_Pre_load(step: int, Pre_value: list, step_interval: int):
    initial_load = Pre_value[0]
    max_load = Pre_value[1]
    step_load = Pre_value[2]
    load_increases = step // step_interval
    current_load = initial_load + load_increases * step_load
    pre_load = min(current_load, max_load)
    return pre_load


def get_loss_weight(step: int, loss_weight: list, step_interval: int):
    initial_weight = loss_weight[0]
    max_weight = loss_weight[1]
    step_weight = loss_weight[2]
    weight_increases = step // step_interval
    current_weight = initial_weight + weight_increases * step_weight
    loss_weight = min(current_weight, max_weight)
    return loss_weight


def _load_checkpoint_to_model(model, ckpt_path: str, load_latent_vectors: bool):
    checkpoint = torch.load(ckpt_path, map_location=dev)
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if not load_latent_vectors and 'latent_vectors' in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != 'latent_vectors'}

    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []

    for key, value in state_dict.items():
        if key not in model_state:
            continue

        target = model_state[key]
        if target.shape == value.shape:
            filtered_state[key] = value
            continue

        # Adapt first layer when checkpoint has xyz-only input, while current model uses xyz + latent.
        if key == 'layers.0.weight' and value.ndim == 2 and target.ndim == 2:
            adapted = target.clone()
            out_min = min(adapted.shape[0], value.shape[0])
            in_min = min(adapted.shape[1], value.shape[1])
            adapted[:out_min, :in_min] = value[:out_min, :in_min]
            filtered_state[key] = adapted
            print(f"Adapted {key}: {tuple(value.shape)} -> {tuple(target.shape)}")
            continue

        skipped_keys.append((key, tuple(value.shape), tuple(target.shape)))

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if skipped_keys:
        print(f"Skipped shape-mismatched keys: {skipped_keys}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")


def _freeze_except_latent_vectors(model):
    for name, param in model.named_parameters():
        param.requires_grad = (name == 'latent_vectors')

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {trainable_params}")



dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    torch.manual_seed(cfg.seed)

    data = Dataset(data_path=cfg.mesh_path, data_num=cfg.data_num, model_scale=cfg.model_scale)
    dom = data.domain()
    bc_Dir = data.bc_Dirichlet(cfg.Dir_marker)
    bc_Pre = data.bc_Pressure(cfg.Pre_marker)
    bc_Sym = data.bc_Symmetry(cfg.Sym_marker)

    dem = ResNet(cfg.input_size, cfg.hidden_size, cfg.output_size, cfg.depth, cfg.data_num, cfg.latent_dim).to(dev)
    start_epoch = 0

    if cfg.pretrained_model_path:
        if not os.path.exists(cfg.pretrained_model_path):
            raise FileNotFoundError(f"pretrained_model_path not found: {cfg.pretrained_model_path}")
        _load_checkpoint_to_model(dem, cfg.pretrained_model_path, cfg.load_pretrained_latent_vectors)

    if cfg.train_latent_only:
        _freeze_except_latent_vectors(dem)

    dem.train()

    start_time = time.time()
    losses = []
    epoch_num = cfg.epoch_num
    learning_rate = cfg.lr

    if cfg.train_latent_only:
        optimizer = torch.optim.Adam([dem.latent_vectors], lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(dem.parameters(), lr=learning_rate)

    lr_history = []

    if cfg.lr_scheduler == 'Cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)
        print(f"train: model_scale={cfg.model_scale}, Net={cfg.depth}x{cfg.input_size}-{cfg.hidden_size}-{cfg.output_size}({cfg.latent_dim}), lr={learning_rate:.0e}, scheduler={cfg.lr_scheduler}{scheduler.T_max}x{cfg.eta_min}, load={cfg.Pre_value[0]:.0e}, {cfg.Pre_value[2]:.0e}, {cfg.Pre_step_interval:.0e}, weight={cfg.loss_weight[0]:.0e}, {cfg.loss_weight[2]:.0e}, {cfg.weight_step_interval:.0e}, latent_only={cfg.train_latent_only}")
    elif cfg.lr_scheduler == 'Exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
        print(f"train: model_scale={cfg.model_scale}, Net={cfg.depth}x{cfg.input_size}-{cfg.hidden_size}-{cfg.output_size}({cfg.latent_dim}), lr={learning_rate:.0e}, scheduler={cfg.lr_scheduler}{scheduler.gamma}, load={cfg.Pre_value[0]:.0e}, {cfg.Pre_value[2]:.0e}, {cfg.Pre_step_interval:.0e}, weight={cfg.loss_weight[0]:.0e}, {cfg.loss_weight[2]:.0e}, {cfg.weight_step_interval:.0e}, latent_only={cfg.train_latent_only}")
    else:
        raise ValueError(f"Unsupported lr_scheduler: {cfg.lr_scheduler}")

    tqdm_epoch = tqdm(range(start_epoch, epoch_num), desc='epoches', colour='red', dynamic_ncols=True)
    for epoch in range(start_epoch, epoch_num):
        Pre_load = get_Pre_load(epoch, cfg.Pre_value, cfg.Pre_step_interval)
        loss_weight = get_loss_weight(epoch, cfg.loss_weight, cfg.weight_step_interval)

        total_loss = 0
        total_eloss = 0
        total_bloss = 0
        optimizer.zero_grad()

        for i in range(cfg.data_num):
            loss = Loss(dem)
            loss_value, energy_loss, boundary_loss = loss.loss_function(i, dom[i], bc_Dir[i], bc_Pre[i], bc_Sym[i], Pre_load, loss_weight)
            total_loss += loss_value / cfg.data_num
            total_eloss += energy_loss / cfg.data_num
            total_bloss += boundary_loss / cfg.data_num

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(total_loss.item())
        lr_history.append(optimizer.param_groups[0]['lr'])

        tqdm_epoch.update()
        tqdm_epoch.set_postfix({
            'loss': '{:.5e}'.format(losses[-1]),
            'eloss': '{:.5e}'.format(total_eloss),
            'bloss': '{:.5e}'.format(total_bloss),
            'lr': '{:.1e}'.format(lr_history[-1]),
            'p': '{:.1e}'.format(Pre_load),
            'w': '{:.1e}'.format(loss_weight)
        })

        if epoch % 5000 == 0:
            os.makedirs(cfg.model_save_path, exist_ok=True)
            torch.save(dem.state_dict(), f"{cfg.model_save_path}/dem_epoch{epoch}.pth")
            with open(f"{cfg.model_save_path}/loss_middle_seed{cfg.seed}.txt", 'w') as f:
                f.write('\n'.join(map(str, losses)) + '\n')

    os.makedirs(cfg.model_save_path, exist_ok=True)
    torch.save(dem.state_dict(), f"{cfg.model_save_path}/dem_epoch{epoch_num}.pth")
    plt.savefig(f"{cfg.model_save_path}/training_curve_epoch{epoch_num}.png")
    with open(f"{cfg.model_save_path}/loss_epoch{epoch_num}_seed{cfg.seed}.txt", 'w') as f:
        f.write('\n'.join(map(str, losses)) + '\n')
    with open(f"{cfg.model_save_path}/lr_epoch{epoch}_seed{cfg.seed}.txt", 'w') as f:
        f.write('\n'.join(map(str, lr_history)) + '\n')

    end_time = time.time() - start_time
    print('End time: %.5f' % end_time)
    print('Training finished. Results saved in %s' % cfg.model_save_path)

