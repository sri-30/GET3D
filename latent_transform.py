import torch
import numpy as np
import copy
import pickle
import time

from clip_utils.clip_loss import CLIPLoss
from training_utils.training_utils import get_lr
from get3d_utils import constructGenerator, eval_get3d_single

class TransformLatent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def preprocess_rgb(array):
    lo, hi = -1, 1
    img = array
    img = img.transpose(1, 3)
    img = img.transpose(1, 2)
    img = (img - lo) * (255 / (hi - lo))
    img.clip(0, 255)
    return img

def train_eval(G, data_geo_z, data_tex_z, text_prompt, n_epochs=5, lmbda_1=0.0015, lmbda_2=0.0015, edit_geo=True, edit_tex=True):
    model_geo = TransformLatent().to('cuda')
    #model_tex = TransformLatent().to('cuda')

    model_geo.train()
    #model_tex.train()

    learning_rate_geo = 1e-3
    optimizer_geo = torch.optim.Adam(model_geo.parameters(), lr=learning_rate_geo)

    # learning_rate_tex = 1e-3
    # optimizer_tex = torch.optim.Adam(model_tex.parameters(), lr=learning_rate_tex)
   
    with torch.no_grad():
        g_ema = copy.deepcopy(G).eval()
        c = torch.ones(1, device='cuda')
        img_original = eval_get3d_single(g_ema, data_geo_z, data_tex_z, c)
    
    #clip_loss = CLIPLoss(text_prompt=text_prompt, target_type='pae', clip_pae_args={'original_image': img_original, 'power': 5.0, 'clip_target': 'paePCA'})
    clip_loss = CLIPLoss(text_prompt=text_prompt, target_type='directional', clip_pae_args={'original_image': img_original, 'source_text': 'car'})
    #clip_loss = CLIPLoss(text_prompt)

    original_latents = (data_geo_z.detach().cpu(), data_tex_z.detach().cpu())
    edited_latents = []
    res_loss = []
    edited_images = []

    min_latent = None
    min_loss = float('inf')

    save_n = 100
    n_save = int(n_epochs)/save_n

    for i in range(n_epochs):
        print(i)
        geo_z = data_geo_z.detach()
        tex_z = data_tex_z.detach()

        g_ema = copy.deepcopy(G).eval()

        # Transform latents with model
        geo_z.requires_grad = True
        tex_z.requires_grad = True
        
        if edit_geo:
            geo_z_edited = model_geo(geo_z).reshape(1, 512)
        else:
            geo_z_edited = geo_z
        # if edit_tex:
        #     tex_z_edited = model_tex(tex_z).reshape(1, 512)
        # else:
        tex_z_edited = tex_z

        # Get output of GET3D on latents
        c = torch.ones(1, device='cuda')
        output = eval_get3d_single(g_ema, geo_z_edited, tex_z_edited, c)

        cur_output = output.detach()

        # Get CLIP Loss
        loss_clip = clip_loss(output[0])

        # Control similarity to original latents
        loss_geo = 0
        loss_tex = 0

        if edit_geo:
            loss_geo = lmbda_1 * ((geo_z_edited - geo_z) ** 2).sum()
        # if edit_tex:
        #     loss_tex = lmbda_2 * ((tex_z_edited - tex_z) ** 2).sum()

        loss = loss_clip # + loss_geo # + loss_tex
        
        # Backpropagation
        loss.backward()

        with torch.no_grad():
            if loss.item() < min_loss:
                min_loss = loss.item()
                min_latent = (geo_z_edited.detach().cpu(), tex_z_edited.detach().cpu())

            res_loss.append((loss.item(), 0, loss_clip.item()))

        t = i / n_epochs
        lr_geo = get_lr(t, learning_rate_geo)
        #lr_tex = get_lr(t, learning_rate_tex)

        if edit_geo:
            optimizer_geo.step()
            optimizer_geo.zero_grad()
            optimizer_geo.param_groups[0]['lr'] = lr_geo
        
        # if edit_tex:
        #     optimizer_tex.step()
        #     optimizer_tex.zero_grad()
        #     optimizer_tex.param_groups[0]['lr'] = lr_tex
        
        if i % n_save == 0:
            edited_images.append(cur_output.cpu())
        
        # edited_latents.append((geo_z_edited.detach().cpu(), tex_z_edited.detach().cpu()))
    
    return original_latents, edited_latents, res_loss, min_latent, edited_images

if __name__ == "__main__":
    c = None
    with open('test.pickle', 'rb') as f:
        c = pickle.load(f)

    G_ema = constructGenerator(**c)

    # Parameters
    random_seed = 0
    lmbda_1 = 0.001
    lmbda_2 = 0.1
    text_prompt= 'Sports Car'
    n_epochs = 500

    torch.manual_seed(random_seed)

    z = torch.randn([1, 512], device='cuda')  # random code for geometry
    tex_z = torch.randn([1, 512], device='cuda')  # random code for texture

    original, edited, loss, min_latent, edited_images = train_eval(G_ema, z, tex_z, text_prompt, n_epochs=n_epochs, lmbda_1=lmbda_1, lmbda_2=lmbda_2, edit_tex=False)

    print(loss)
    print(min(loss))

    result = []

    with torch.no_grad():
        G_ema.eval()
        img_original = eval_get3d_single(G_ema, original[0].to('cuda'), original[1].to('cuda'), torch.ones(1, device='cuda')).cpu()
        img_edited = eval_get3d_single(G_ema, min_latent[0].to('cuda'), min_latent[1].to('cuda'), torch.ones(1, device='cuda')).cpu()
        result.append({'Original': img_original, 'Edited': edited_images[-1], 'Loss': loss, 'Original Latent': original, 'Edited Latent': min_latent, 'Edited Images': edited_images})
    with open(f'latent_transform_adam_results_pae/output_img_{random_seed}_{lmbda_1}_{lmbda_2}_{text_prompt}_{n_epochs}_{time.time()}.pickle', 'wb') as f:
        pickle.dump(result, f)
    
    
