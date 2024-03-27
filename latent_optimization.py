import torch
import numpy as np
import copy
import pickle
import time

from clip_utils.clip_loss_nada import CLIPLoss
from training_utils.training_utils import get_lr
from get3d_utils import constructGenerator, eval_get3d_single, eval_get3d_weights, eval_get3d_angles

def train_eval(G, data_geo_z, data_tex_z, text_prompt, n_epochs=5, lmbda_1=0.0015, lmbda_2=0.0015, edit_geo=True, edit_tex=True, intermediate_space=False):
    camera_idx = [4, 8, 12]
    
    with torch.no_grad():
        g_ema = copy.deepcopy(G).eval()
        c = torch.ones(1, device='cuda')
        imgs_original = eval_get3d_angles(g_ema, data_geo_z.clone(), data_tex_z.clone(), camera_idx=camera_idx, intermediate_space=False)
        img_original = imgs_original[0].detach().clone()

    clip_loss = CLIPLoss('car', text_prompt, corpus=['Sports Car', 'SUV', 'Hatchback', 'Sedan'], original_images=imgs_original, aux_string='')

    edited_latents = []
    res_loss = []
    edited_images = []

    min_latent = None
    min_loss = float('inf')

    save_n = 100
    n_save = int(n_epochs)/save_n

    g_ema = copy.deepcopy(G).eval()

    learning_rate = 1e-2
    if intermediate_space:
        ws_geo = g_ema.mapping_geo(geo_z, None)
        ws_tex = g_ema.mapping(tex_z, None)
        data_geo = ws_geo.detach()
        data_tex = ws_tex.detach()
        latent_geo = ws_geo.detach().clone()
        latent_tex = ws_tex.detach().clone()
        for layer in latent_geo:
            layer.requires_grad_(True)
        for layer in latent_tex:
            layer.requires_grad_(True)
        z_optim = torch.optim.Adam([layer for layer in latent_geo] + [layer for layer in latent_tex], lr=learning_rate)
    else:
        data_geo = data_geo_z.detach().clone()
        data_tex = data_tex_z.detach().clone()
        latent_geo = data_geo_z.detach().clone()
        latent_tex = data_tex_z.detach().clone()
        z_optim = torch.optim.Adam([latent_geo, latent_tex], lr=learning_rate)

    original_latents = (latent_geo.detach().clone().cpu(), latent_tex.detach().clone().cpu())

    latent_geo.requires_grad_(True)
    latent_tex.requires_grad_(False)

    for i in range(n_epochs):
        print(i)

        # Get output of GET3D on latents
        #output = eval_get3d_weights(g_ema, geo_z, tex_z, 0)
        output = eval_get3d_angles(g_ema, latent_geo, latent_tex, camera_idx=camera_idx, intermediate_space=intermediate_space)

        # Get CLIP Loss
        # loss_clip = clip_loss(imgs_original, output)
        # loss_clip = clip_loss.projection_embedding_loss(output)
        loss_clip = clip_loss.global_loss(output)
        # loss_clip = clip_loss.projection_augmentation_loss(output)

        # Control similarity to original latents
        loss_geo = 0
        loss_tex = 0

        if edit_geo:
            loss_geo = lmbda_1 * ((latent_geo - data_geo) ** 2).sum()
        if edit_tex:
            loss_tex = lmbda_2 * ((latent_tex - data_tex) ** 2).sum()

        loss = loss_clip + loss_geo + loss_tex # + lmbda_geo_deviation * (geo_z ** 2).sum() + lmbda_tex_deviation * (tex_z ** 2).sum()
        # Backpropagation
        

        # lr_geo = get_lr(t, learning_rate_geo)
        # lr_tex = get_lr(t, learning_rate_tex)
        
        loss.backward()
        z_optim.step()
        z_optim.zero_grad()

        # t = i / n_epochs
        # lr = get_lr(t, learning_rate)
        # z_optim.param_groups[0]['lr'] = lr

        # if edit_geo:
        #     optimizer_geo.step()
        #     optimizer_geo.zero_grad()
        # #    optimizer_geo.param_groups[0]['lr'] = lr_geo
        
        # if edit_tex:
        #     optimizer_tex.step()
        #     optimizer_tex.zero_grad()
        # #    optimizer_tex.param_groups[0]['lr'] = lr_tex

        with torch.no_grad():
            if loss.item() < min_loss:
                min_loss = loss.item()
                min_latent = (latent_geo.detach().cpu(), latent_tex.detach().cpu())

            res_loss.append((loss.item(), loss_geo.item(), loss_clip.item()))

            if i % n_save == 0:
                cur_output = output.detach().clone()
                edited_images.append(cur_output[0].cpu().unsqueeze(0))

            edited_latents.append((latent_geo.detach().cpu(), latent_tex.detach().cpu()))
    
    return original_latents, edited_latents, res_loss, min_latent, edited_images

if __name__ == "__main__":
    with open('test.pickle', 'rb') as f:
        c = pickle.load(f)
    
    G_ema = constructGenerator(**c)

    # Parameters
    random_seed = 0
    lmbda_1 = 0.005
    lmbda_2 = 0.01
    text_prompt= 'SUV'
    n_epochs = 200
    intermediate_space=False

    torch.manual_seed(random_seed)

    z = torch.randn([1, 512], device='cuda')  # random code for geometry
    tex_z = torch.randn([1, 512], device='cuda')  # random code for texture

    original, edited, loss, min_latent, edited_images = train_eval(G_ema, z, tex_z, text_prompt, n_epochs=n_epochs, lmbda_1=lmbda_1, lmbda_2=lmbda_2, intermediate_space=intermediate_space)

    print(loss)
    print(min(loss))

    result = []

    with torch.no_grad():
        G_ema.eval()
        img_original = eval_get3d_angles(G_ema, original[0].to('cuda'), original[1].to('cuda'), intermediate_space=intermediate_space)[0].unsqueeze(0).cpu()
        img_edited = eval_get3d_angles(G_ema, min_latent[0].to('cuda'), min_latent[1].to('cuda'), intermediate_space=intermediate_space)[0].unsqueeze(0).cpu()
        result.append({'Original': img_original, 'Edited': img_edited, 'Loss': loss, 'Original Latent': original, 'Edited Latent': min_latent, 'Edited Images': edited_images})
    with open(f'latent_transform_adam_results_pae/output_img_{random_seed}_{lmbda_1}_{lmbda_2}_{text_prompt}_{n_epochs}_{time.time()}.pickle', 'wb') as f:
        pickle.dump(result, f)
    
    images_latents = {'Original Image': img_original, 'Edited Image': img_edited, 'Original Latent': original, 'Edited Latent': min_latent}
    with open(f'latents_saved/output_{random_seed}_{text_prompt}.pickle', 'wb') as f:
        pickle.dump(images_latents, f)
    
