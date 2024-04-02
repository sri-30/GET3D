import torch
import numpy as np
import copy
import pickle
import time

from clip_utils.clip_loss_nada import CLIPLoss
from training_utils.training_utils import get_lr
from get3d_utils import constructGenerator, eval_get3d_single, eval_get3d_angles, eval_get3d_weights_ws, eval_nada, determine_opt_layers, unfreeze_generator_layers, freeze_generator_layers
from torch_utils import misc

def preprocess_rgb(array):
    lo, hi = -1, 1
    img = array
    img = img.transpose(1, 3)
    img = img.transpose(1, 2)
    img = (img - lo) * (255 / (hi - lo))
    img.clip(0, 255)
    return img

def train_eval(G, data_geo_z, data_tex_z, text_prompt, n_epochs=5, lmbda_1=0.0015, lmbda_2=0.0015, edit_geo=True, edit_tex=True):
    g_ema_frozen = copy.deepcopy(G)
    g_ema_train = copy.deepcopy(G)
    
    g_ema_frozen.update_w_avg()
    g_ema_train.update_w_avg()
    
    g_ema_frozen.requires_grad_(False)
    g_ema_frozen.eval()

    g_ema_train.mapping.requires_grad_(False)
    g_ema_train.mapping_geo.requires_grad_(False)
    g_ema_train.train()

    with torch.no_grad():
        grid_c = torch.ones(1, device='cuda')
        tex_z = data_tex_z.cuda()
        geo_z = data_geo_z.cuda()

        data_ws = g_ema_frozen.mapping(tex_z, grid_c, update_emas=False)
        data_ws_geo = g_ema_frozen.mapping_geo(
            geo_z, grid_c,
            update_emas=False)
        img_original = eval_get3d_single(g_ema_frozen, geo_z, tex_z, grid_c)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(g_ema_train.parameters(), 
                                lr=learning_rate,
                                betas=(0.9, 0.99))

    #clip_loss = CLIPLoss(text_prompt=text_prompt, target_type='PAE', clip_pae_args={'original_image': img_original, 'power': 0.3, 'clip_target': 'PCA+'})
    #clip_loss = CLIPLoss(text_prompt)
    clip_loss = CLIPLoss(source_text='car', target_text=text_prompt, corpus=['Sports Car', 'SUV', 'Hatchback', 'Sedan'], aux_string='')
    res_loss = []
    edited_images = []

    min_loss = float('inf')

    save_n = 100
    n_batch = 3
    n_save = int(n_epochs)/save_n

    camera_idx_ = [4]
    camera_idx_epoch = {0: [4, 7], 1: [4, 7], 2: [6, 8], 3: [4, 7]}

    n_training_samples = 500

    train_z_geo = torch.randn(n_training_samples, G.z_dim, device='cuda')
    train_z_tex = torch.randn(n_training_samples, G.z_dim, device='cuda')

    z_geo_split = torch.split(train_z_geo, n_batch, dim=0)
    z_tex_split = torch.split(train_z_tex, n_batch, dim=0)

    for i in range(n_epochs):
        print(f'Epoch: {i}')
        loss = 0


        for j in range(len(z_geo_split)):
            print(f'Batch: {j}')

            # Layer freezing
            unfreeze_generator_layers(g_ema_train, [], [])
            topk_idx_tex, topk_idx_geo = determine_opt_layers(g_ema_frozen, g_ema_train, clip_loss)
            freeze_generator_layers(g_ema_train)
            unfreeze_generator_layers(g_ema_train, topk_idx_tex, topk_idx_geo)

            loss = 0

            for k in range(len(z_geo_split[j])):

                g_ema_train.train()

                z_geo = z_geo_split[j][k].unsqueeze(0)
                z_tex = z_tex_split[j][k].unsqueeze(0)

                with torch.no_grad():
                    ws_geo = g_ema_frozen.mapping_geo(z_geo, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)
                    ws_tex = g_ema_frozen.mapping(z_tex, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)
                
                #ws_geo.requires_grad_(True)
                #ws_tex.requires_grad_(True)

                # Get output of GET3D on latents
                with torch.no_grad():
                    frozen_img = eval_get3d_angles(g_ema_frozen, ws_geo, ws_tex, camera_idx=camera_idx_, intermediate_space=True)
                
                trainable_img = eval_get3d_angles(g_ema_train, ws_geo, ws_tex, camera_idx=camera_idx_, intermediate_space=True)

                # with torch.no_grad():
                #     frozen_img = eval_nada(g_ema_frozen, g_ema_frozen,ws=ws_tex, ws_geo=ws_geo, n_cameras=2)

                # trainable_img = eval_nada(g_ema_train, g_ema_frozen,ws=ws_tex, ws_geo=ws_geo, n_cameras=2)

                loss += clip_loss(frozen_img, trainable_img).sum()
                # loss += clip_loss.projection_augmentation_loss_nada(frozen_img, trainable_img)

            torch.nn.utils.clip_grad_norm_(g_ema_frozen.parameters(), -1)

            loss /= n_batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                g_ema_train.eval()
                img_save = eval_get3d_weights_ws(g_ema_train, data_ws_geo, data_ws, 0)
                loss_ = clip_loss(img_original, img_save).sum()
                # loss_ = clip_loss.projection_augmentation_loss_nada(img_original, img_save)
                res_loss.append((loss_.item(), 0, 0))
                # edited_images.append(img_save.cpu())
            torch.cuda.empty_cache()

    return g_ema_train, res_loss, edited_images

if __name__ == "__main__":


    with open('test.pickle', 'rb') as f:
            c = pickle.load(f)
    print(c)
    G_ema = constructGenerator(**c)

    # Parameters
    random_seed = 2
    lmbda_1 = 0.001
    lmbda_2 = 0.1
    text_prompt= 'Lamborghini'
    n_epochs = 4

    torch.manual_seed(random_seed)

    z = torch.randn([1, 512], device='cuda')  # random code for geometry
    tex_z = torch.randn([1, 512], device='cuda')  # random code for texture

    G_new, loss, edited_images = train_eval(G_ema, z, tex_z, text_prompt, n_epochs=n_epochs, lmbda_1=lmbda_1, lmbda_2=lmbda_2, edit_tex=False)

    print(loss)
    print(min(loss))

    result = []

    with torch.no_grad():
        G_new.eval()
        img_original = eval_get3d_single(G_ema, z, tex_z, torch.ones(1, device='cuda')).cpu()
        img_edited = eval_get3d_single(G_new, z, tex_z, torch.ones(1, device='cuda')).cpu()
        result.append({'Original': img_original, 'Edited': img_edited, 'Loss': loss, 'Edited Images': edited_images})
    with open(f'weight_transform_results/output_img_{random_seed}_{lmbda_1}_{lmbda_2}_{text_prompt}_{n_epochs}_{time.time()}.pickle', 'wb') as f:
        pickle.dump(result, f)
    
    
