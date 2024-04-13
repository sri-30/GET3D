import torch
import numpy as np
import copy
import pickle
import time

from clip_utils.clip_loss_nada import CLIPLoss
from training_utils.training_utils import get_lr
from get3d_utils import constructGenerator, save_textured_mesh, eval_get3d_angles, generate_random_camera, generate_rotate_camera_list
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./runs")

def train_eval(G, data_geo_z, data_tex_z, text_prompt, n_epochs=5, lmbda_1=0.0015, lmbda_2=0.0015, edit_geo=True, edit_tex=True, intermediate_space=False, loss_type = 'global', corpus_type='body_type'):
    camera_list = generate_rotate_camera_list()
    
    with torch.no_grad():
        g_ema = copy.deepcopy(G).eval()
        c = torch.ones(1, device='cuda')
        imgs_original = eval_get3d_angles(g_ema, data_geo_z.clone(), data_tex_z.clone(), cameras=camera_list, intermediate_space=False)

    if corpus_type == 'body_type':
        corpus = ['Sports Car', 'SUV', 'Hatchback', 'Sedan']
    elif corpus_type == 'textures':
        corpus = ["Paint", "Glass", "Chrome", "Rubber", "Leather", "Plastic", "Carbon Fiber", "Fabric", "Metal", "Rust", "Dirt", "Decal", "Reflective Surfaces", "Ice", "Wood", "Neon", "Matte Black", "Carbon Ceramic"]
    else:
        raise NotImplementedError


    clip_loss = CLIPLoss('car', text_prompt, corpus=corpus, original_images=imgs_original, aux_string='')

    edited_latents = []
    res_loss = []
    edited_images = []

    min_latent = None
    min_loss = float('inf')

    save_n = 100
    n_save = int(n_epochs)/save_n

    g_ema = copy.deepcopy(G).eval()

    learning_rate = 5e-2
    if intermediate_space:
        with torch.no_grad():
            data_geo = g_ema.mapping_geo(data_geo_z, c=torch.ones(1, device='cuda'), truncation_psi=0.7, update_emas=False).detach()
            data_tex = g_ema.mapping(data_tex_z, c=torch.ones(1, device='cuda'), truncation_psi=0.7, update_emas=False).detach()
        latent_geo = data_geo[0][0].detach().unsqueeze(0).clone()
        latent_tex = data_tex[0][0].detach().unsqueeze(0).clone()
    else:
        data_geo = data_geo_z.detach().clone()
        data_tex = data_tex_z.detach().clone()
        latent_geo = data_geo_z.detach().clone()
        latent_tex = data_tex_z.detach().clone()

    original_latents = (data_geo.detach().clone().cpu(), data_tex.detach().clone().cpu())

    latent_geo.requires_grad_(True)
    latent_tex.requires_grad_(True)

    z_optim = torch.optim.Adam([latent_geo, latent_tex], lr=learning_rate, betas=(0.9, 0.99))

    for i in range(n_epochs):
        print(i)

        cameras = generate_random_camera(1, n_views=10)
        cameras = cameras.unsqueeze(0)
        cameras = cameras.transpose(0, 2)
        # cameras = generate_rotate_camera_list()

        with torch.no_grad():
            output_original = eval_get3d_angles(g_ema,  data_geo_z, data_tex_z, cameras=cameras,  intermediate_space=False)

        # Get output of GET3D on latents
        #output = eval_get3d_weights(g_ema, geo_z, tex_z, 0)
        if intermediate_space:
            output_edited = eval_get3d_angles(g_ema, latent_geo.unsqueeze(0).repeat(1, 22, 1), 
                                       latent_tex.unsqueeze(0).repeat(1, 9, 1),
                                       cameras=cameras, intermediate_space=intermediate_space)
        else:
            output_edited = eval_get3d_angles(g_ema, latent_geo, latent_tex, cameras=cameras, intermediate_space=intermediate_space)
        # Get CLIP Loss
        # loss_clip = clip_loss.directional_projection_loss(output_edited, output_edited).mean()
        # loss_clip = clip_loss(output_edited, output_edited)
        # loss_clip = clip_loss.projection_embedding_loss(output_edited)
        if loss_type == 'global':
            loss_clip = clip_loss.global_loss(output_edited)
        elif loss_type == 'pae':
            loss_clip = clip_loss.projection_augmentation_loss_nada(output_original, output_edited, power=8)
        elif loss_type == 'directional':
            loss_clip = clip_loss.directional_loss(output_original, output_edited)
        else:
            raise NotImplementedError
        # loss_clip = clip_loss(output_original, output_edited).mean()

        # Control similarity to original latents
        loss_geo = 0
        loss_tex = 0

        if edit_geo:
            loss_geo = lmbda_1 * ((latent_geo - data_geo) ** 2).sum()
        if edit_tex:
            loss_tex = lmbda_2 * ((latent_tex - data_tex) ** 2).sum()

        loss = loss_clip + loss_geo + loss_tex # + lmbda_geo_deviation * (geo_z ** 2).sum() + lmbda_tex_deviation * (tex_z ** 2).sum()
 
        # Backpropagation
        loss.backward()

        writer.add_scalar("Loss/train", loss, i)

        # Update Optimizer
        z_optim.step()
        z_optim.zero_grad()

        with torch.no_grad():
            if loss.item() < min_loss:
                min_loss = loss.item()
                min_latent = (latent_geo.detach().cpu(), latent_tex.detach().cpu())

            res_loss.append((loss.item(), loss_geo.item(), loss_clip.item()))

            if i % n_save == 0:
                cur_output = output_edited.detach().clone()
                edited_images.append(cur_output[4].cpu().unsqueeze(0))

            edited_latents.append((latent_geo.detach().cpu(), latent_tex.detach().cpu()))
    if intermediate_space:
        return original_latents, edited_latents, res_loss, (min_latent[0].unsqueeze(0).repeat(1, 22, 1), min_latent[1].unsqueeze(0).repeat(1, 9, 1)), edited_images
    else:
        return original_latents, edited_latents, res_loss, min_latent, edited_images

if __name__ == "__main__":
    import sys

    print(sys.argv)
    _, random_seed_, text_prompt_, loss_type_, lmbda_1_, lmbda_2_ = sys.argv

    with open('test.pickle', 'rb') as f:
        c = pickle.load(f)
    
    G_ema = constructGenerator(**c)

    # Parameters
    random_seed = int(random_seed_)
    lmbda_1 = float(lmbda_1_)
    lmbda_2 = float(lmbda_2_)
    text_prompt= text_prompt_
    n_epochs = 100
    intermediate_space=True

    torch.manual_seed(random_seed)

    z = torch.randn([1, 512], device='cuda')  # random code for geometry
    tex_z = torch.randn([1, 512], device='cuda')  # random code for texture

    original, edited, loss, min_latent, edited_images = train_eval(G_ema, z, tex_z, text_prompt, n_epochs=n_epochs, lmbda_1=lmbda_1, lmbda_2=lmbda_2, intermediate_space=intermediate_space, loss_type=loss_type_)

    writer.close()

    print(loss)
    print(min(loss))

    result = []

    with torch.no_grad():
        G_ema.eval()
        cameras = generate_rotate_camera_list()
        img_original = eval_get3d_angles(G_ema, original[0].to('cuda'), original[1].to('cuda'), cameras=cameras, intermediate_space=intermediate_space).cpu()
        img_edited = eval_get3d_angles(G_ema, min_latent[0].to('cuda'), min_latent[1].to('cuda'), cameras=cameras, intermediate_space=intermediate_space).cpu()
        save_textured_mesh(G_ema, original[0].to('cuda'), original[1].to('cuda'), f'meshes_saved/output_{random_seed}_original.obj')
        save_textured_mesh(G_ema, min_latent[0].to('cuda'), min_latent[1].to('cuda'), f'meshes_saved/output_{random_seed}_{text_prompt_}_{loss_type_}.obj')
        # result.append({'Original': img_original, 'Edited': img_edited, 'Loss': loss, 'Original Latent': original, 'Edited Latent': min_latent, 'Edited Images': edited_images})
    # with open(f'latent_transform_adam_results_pae/output_img_{random_seed}_{lmbda_1}_{lmbda_2}_{text_prompt}_{n_epochs}_{time.time()}.pickle', 'wb') as f:
    #     pickle.dump(result, f)
    
    images_latents = {'Original Image': img_original, 'Edited Image': img_edited, 'Original Latent': original, 'Edited Latent': min_latent, 'Loss': loss}
    with open(f'latents_saved/output_{random_seed}_{text_prompt}_{loss_type_}.pickle', 'wb') as f:
        pickle.dump(images_latents, f)
    
