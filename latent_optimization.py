import torch
import numpy as np
import copy
import pickle

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
    model_tex = TransformLatent().to('cuda')

    model_geo.train()
    model_tex.train()

    geo_z = data_geo_z.detach()
    tex_z = data_tex_z.detach()

    geo_z.requires_grad = True
    tex_z.requires_grad = True

    learning_rate_geo = 0.1
    optimizer_geo = torch.optim.Adam([geo_z], lr=learning_rate_geo)

    learning_rate_tex = 0.1
    optimizer_tex = torch.optim.Adam([tex_z], lr=learning_rate_tex)

    with torch.no_grad():
        g_ema = copy.deepcopy(G).eval()
        c = torch.ones(1, device='cuda')
        img_original = eval_get3d_single(g_ema, geo_z, tex_z, c)
    
    # clip_loss = CLIPLoss(text_prompt, target_type='PAE', clip_pae_args={'original_image': img_original})
    clip_loss = CLIPLoss(text_prompt)

    original_latents = (data_geo_z.detach().cpu(), data_tex_z.detach().cpu())
    edited_latents = []
    res_loss = []

    min_latent = None
    min_loss = float('inf')

    for i in range(n_epochs):
        print(i)

        g_ema = copy.deepcopy(G).eval()

        # Get output of GET3D on latents
        c = torch.ones(1, device='cuda')
        output = eval_get3d_single(g_ema, geo_z, tex_z, c)

        cur_output = output.detach()

        # Get CLIP Loss
        loss_clip = clip_loss(output[0])

        # Control similarity to original latents
        loss_geo = 0
        loss_tex = 0

        if edit_geo:
            loss_geo = lmbda_1 * ((geo_z - data_geo_z) ** 2).sum()
        if edit_tex:
            loss_tex = lmbda_2 * ((tex_z - data_tex_z) ** 2).sum()

        loss = loss_clip + loss_geo + loss_tex
        
        # Backpropagation
        loss.backward()

        if abs(loss.item()) < min_loss:
            min_loss = abs(loss.item())
            min_latent = (geo_z.detach().cpu(), tex_z.detach().cpu())

        res_loss.append(loss.item())

        t = i / n_epochs
        lr = get_lr(t, 0.1)

        if edit_geo:
            optimizer_geo.step()
            optimizer_geo.zero_grad()
            optimizer_geo.param_groups[0]['lr'] = lr
        
        if edit_tex:
            optimizer_tex.step()
            optimizer_tex.zero_grad()
            optimizer_tex.param_groups[0]['lr'] = lr

        edited_latents.append((geo_z.detach().cpu(), tex_z.detach().cpu()))
    
    return original_latents, edited_latents, res_loss, min_latent

if __name__ == "__main__":
    c = None
    with open('test.pickle', 'rb') as f:
        c = pickle.load(f)

    G_ema = constructGenerator(**c)

    torch.manual_seed(4)

    z = torch.randn([1, 512], device='cuda')  # random code for geometry
    tex_z = torch.randn([1, 512], device='cuda')  # random code for texture

    original, edited, loss, min_latent = train_eval(G_ema, z, tex_z, 'Sports Car', n_epochs=100, lmbda_1=0.01, lmbda_2=0.001)

    print(loss)
    print(min([abs(l) for l in loss]))

    result = []

    with torch.no_grad():
        G_ema.eval()
        img_original = eval_get3d_single(G_ema, original[0].to('cuda'), original[1].to('cuda'), torch.ones(1, device='cuda'))
        img_edited = eval_get3d_single(G_ema, min_latent[0].to('cuda'), min_latent[1].to('cuda'), torch.ones(1, device='cuda'))
        result.append((img_original.cpu(), img_edited.cpu()))
    with open('output_img.pickle', 'wb') as f:
        pickle.dump(result, f)
    
