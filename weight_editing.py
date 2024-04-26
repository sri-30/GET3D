import torch
import numpy as np
import copy
import pickle
import time

from clip_utils.clip_loss_nada import CLIPLoss
from get3d_utils import constructGenerator, eval_get3d_angles, determine_opt_layers, unfreeze_generator_layers, freeze_generator_layers, generate_random_camera, generate_rotate_camera_list

def preprocess_rgb(array):
    lo, hi = -1, 1
    img = array
    img = img.transpose(1, 3)
    img = img.transpose(1, 2)
    img = (img - lo) * (255 / (hi - lo))
    img.clip(0, 255)
    return img

def train_eval(G, text_prompt, n_epochs=5, loss_type='global', clip_grad_norm=None, profiler=None):

    camera_list = generate_rotate_camera_list()[::4]

    g_ema_frozen = copy.deepcopy(G)
    g_ema_train = copy.deepcopy(G)
    
    g_ema_frozen.update_w_avg()
    g_ema_train.update_w_avg()
    
    g_ema_frozen.requires_grad_(False)
    g_ema_frozen.eval()

    g_ema_train.mapping.requires_grad_(False)
    g_ema_train.mapping_geo.requires_grad_(False)
    g_ema_train.train()

    learning_rate = 5e-4
    opt_params = g_ema_train.synthesis.generator.parameters()
    optimizer = torch.optim.Adam(opt_params, 
                                lr=learning_rate,
                                betas=(0.9, 0.99))

    clip_loss = CLIPLoss(source_text='car', target_text=text_prompt, corpus=['Sports Car', 'SUV', 'Hatchback', 'Sedan'], aux_string='')

    n_batch = 5

    n_training_samples = n_samp
    train_z_geo = torch.randn(n_training_samples, G.z_dim, device='cuda')
    train_z_tex = torch.randn(n_training_samples, G.z_dim, device='cuda')

    n_val = 3
    validation_geo = torch.randn(n_val, G.z_dim, device='cuda')
    validation_tex = torch.randn(n_val, G.z_dim, device='cuda')

    with torch.no_grad():
        validation_geo = g_ema_frozen.mapping_geo(validation_geo, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)[:,0,]
        validation_tex = g_ema_frozen.mapping(validation_tex, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)[:,0,]

        train_ws_geo = g_ema_frozen.mapping_geo(train_z_geo, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)[:,0,]
        train_ws_tex = g_ema_frozen.mapping(train_z_tex, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)[:,0,]

    ws_geo_split = torch.split(train_ws_geo, n_batch, dim=0)
    ws_tex_split = torch.split(train_ws_tex, n_batch, dim=0)

    metrics = {'train_loss': [], 'val_loss': [], 'lpips': []}

    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.cuda()

    for i in range(n_epochs):
        print(f'Epoch: {i}')
        for j in range(len(ws_geo_split)):
            if not profiler is None:
                profiler.step()
            print(f'Batch: {j}')

            # Layer freezing
            unfreeze_generator_layers(g_ema_train, [], [])
            topk_idx_tex, topk_idx_geo = determine_opt_layers(g_ema_frozen, g_ema_train, clip_loss)
            freeze_generator_layers(g_ema_train)
            unfreeze_generator_layers(g_ema_train, topk_idx_tex, topk_idx_geo)

            loss_log = 0

            for k in range(len(ws_geo_split[j])):

                g_ema_train.train()

                ws_geo = ws_geo_split[j][k].unsqueeze(0).repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1)
                ws_tex = ws_tex_split[j][k].unsqueeze(0).repeat(1, g_ema_frozen.synthesis.generator.num_ws_tex, 1)

                # Get output of GET3D on latents
                with torch.no_grad():
                    cameras = generate_random_camera(1, n_views=camera_num)
                    cameras = cameras.unsqueeze(0)
                    cameras = cameras.transpose(0, 2)

                    frozen_img = eval_get3d_angles(g_ema_frozen, ws_geo, ws_tex, cameras=cameras, intermediate_space=True)
                
                trainable_img = eval_get3d_angles(g_ema_train, ws_geo, ws_tex, cameras=cameras, intermediate_space=True)

                if loss_type == 'global':
                    loss = clip_loss.global_loss(trainable_img)
                elif loss_type == 'pae':
                    loss = clip_loss.projection_augmentation_loss_nada(frozen_img, trainable_img)
                elif loss_type == 'directional':
                    loss = clip_loss(frozen_img, trainable_img).sum()
                else:
                    raise NotImplementedError
                
                loss.backward()
                with torch.no_grad():
                    loss_log += loss.item()

            if not (clip_grad_norm is None):
                torch.nn.utils.clip_grad_norm_(g_ema_train.synthesis.generator.tri_plane_synthesis.parameters(), clip_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # Validation
            with torch.no_grad():
                metrics['train_loss'].append(loss_log / n_batch)
                g_ema_train.eval()
                validation_img_original = eval_get3d_angles(g_ema_frozen, validation_geo.repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1), validation_tex.repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1), cameras=camera_list, intermediate_space=True)
                validation_img_edited = eval_get3d_angles(g_ema_train, validation_geo.repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1), validation_tex.repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1), cameras=camera_list, intermediate_space=True)
                if loss_type == 'global':
                    loss_ = clip_loss.global_loss(validation_img_edited)
                elif loss_type == 'pae':
                    loss_ = clip_loss.projection_augmentation_loss_nada(validation_img_original, validation_img_edited)
                elif loss_type == 'directional':
                    loss_ = clip_loss(validation_img_original, validation_img_edited)
                    loss_ = loss_.sum() / (len(camera_list))
                else:
                    raise NotImplementedError

                metrics['val_loss'].append(loss_.item())
                metrics['lpips'].append(loss_fn_alex((validation_img_original).to('cuda').clip(-1, 1), (validation_img_edited).to('cuda').clip(-1, 1)).mean().item())
            

    return g_ema_train, metrics

if __name__ == "__main__":
    import sys

    _, random_seed_, text_prompt_, loss_type_, profile_dir = sys.argv

    with open('params.pickle', 'rb') as f:
            g_ema_params = pickle.load(f)

    G_ema = constructGenerator(**g_ema_params)

    # Parameters
    random_seed = random_seed_
    text_prompt= text_prompt_
    n_epochs = 10
    n_samp = 90

    camera_num = 7

    torch.manual_seed(random_seed)

    z = torch.randn([1, 512], device='cuda')  # random code for geometry
    tex_z = torch.randn([1, 512], device='cuda')  # random code for texture

    if profile_dir:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=3, warmup=3, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True,
            with_stack=True) as prof:
            G_new, metrics = train_eval(G_ema, text_prompt, n_epochs=n_epochs, loss_type=loss_type_, profiler=prof)
    else:
        G_new, metrics = train_eval(G_ema, text_prompt, n_epochs=n_epochs, loss_type=loss_type_)

    torch.save(G_new.state_dict(), f'generators_saved/foo_{n_samp}_samples_{text_prompt}_{loss_type_}_generator.pt')

    with open(f'./metrics/foo_weight_transform_{n_samp}_samples_{text_prompt}_{loss_type_}_train_val_loss.pickle', 'wb') as f:
        pickle.dump(metrics, f)
    
    
