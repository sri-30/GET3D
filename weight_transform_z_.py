import torch
import numpy as np
import copy
import pickle
import time

from clip_utils.clip_loss import CLIPLoss
from training_utils.training_utils import get_lr
from get3d_utils import constructGenerator, eval_get3d_single, eval_get3d_weights, eval_get3d_weights_ws
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

        img_original = eval_get3d_weights(g_ema_frozen, geo_z, tex_z, grid_c)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(g_ema_train.parameters(), lr=learning_rate)
    
    #clip_loss = CLIPLoss(text_prompt=text_prompt, target_type='PAE', clip_pae_args={'original_image': img_original, 'power': 0.3, 'clip_target': 'PCA+'})
    clip_loss_global = CLIPLoss(text_prompt=text_prompt)
    clip_loss_directional = CLIPLoss(text_prompt=text_prompt, target_type='directional', clip_pae_args={'original_image': img_original, 'source_text': 'a car'})

    res_loss = []
    edited_images = []

    min_loss = float('inf')

    save_n = 100
    n_save = int(n_epochs)/save_n

    for i in range(n_epochs):
        print(i)

        # z_geo = torch.randn(data_geo_z.shape, device='cuda')
        # z_tex = torch.randn(data_tex_z.shape, device='cuda')
        z_geo = data_geo_z.detach()
        z_tex = data_tex_z.detach()

        # Get output of GET3D on latents
        c = torch.ones(1, device='cuda')
        #g_ema.requires_grad_(True)
        output = eval_get3d_weights(g_ema_train, z_geo, z_tex, c)
        with torch.no_grad():
            img_original = eval_get3d_weights(g_ema_frozen, z_geo, z_tex, c)
        cur_output = output.detach()

        # Get CLIP Loss
        #loss_clip = clip_loss_directional(output[0], original_image=img_original) + clip_loss_global(output[0])
        loss_clip = clip_loss_global(output[0])

        # Backpropagation
        optimizer.zero_grad(set_to_none=False)
        loss_clip.backward()
        #g_ema.requires_grad_(False)

        params = [param for param in g_ema_train.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)
        optimizer.step()
        
        t = i / n_epochs
        # lr = get_lr(t, learning_rate)
        # lr_tex = get_lr(t, learning_rate_tex)

        
        with torch.no_grad():
            res_loss.append((loss_clip.item(), 0, 0))
        # optimizer.param_groups[0]['lr'] = lr
    
        if i % n_save == 0:
            edited_images.append(cur_output.cpu())
        
        # edited_latents.append((geo_z_edited.detach().cpu(), tex_z_edited.detach().cpu()))
    
    return g_ema_train, res_loss, edited_images

if __name__ == "__main__":
    c = None
    with open('test.pickle', 'rb') as f:
        c = pickle.load(f)

    G_ema = constructGenerator(**c)

    # Parameters
    random_seed = 0
    lmbda_1 = 0.001
    lmbda_2 = 0.1
    text_prompt= 'a sports car'
    n_epochs = 2000

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
    
    
