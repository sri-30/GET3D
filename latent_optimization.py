import torch
import numpy as np
import copy
import pickle

from clip_utils.clip_loss import CLIPLoss
from get3d_utils import constructGenerator, eval_get3d_single

class TransformLatent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
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

def train_eval(G, data_geo_z, data_tex_z, text_prompt, n_epochs=5, lmbda_1=0.0015, lmbda_2=0.0015):
    model_geo = TransformLatent().to('cuda')
    model_tex = TransformLatent().to('cuda')

    model_geo.train()
    model_tex.train()

    learning_rate_geo = 1e-3
    optimizer_geo = torch.optim.SGD(model_geo.parameters(), lr=learning_rate_geo)

    learning_rate_tex = 1e-3
    optimizer_tex = torch.optim.SGD(model_tex.parameters(), lr=learning_rate_tex)
    
    loss_fn = CLIPLoss(text_prompt)

    original_latents = (data_geo_z.detach().cpu(), data_tex_z.detach().cpu())
    edited_latents = []
    res_loss = []

    for i in range(n_epochs):
        print(i)
        geo_z = data_geo_z.detach()
        tex_z = data_tex_z.detach()

        g_ema = copy.deepcopy(G).eval()

        # Transform latents with model
        geo_z.requires_grad = True
        tex_z.requires_grad = True

        geo_z_edited = model_geo(geo_z).reshape(1, 512)
        tex_z_edited = model_tex(tex_z).reshape(1, 512)

        # Get output of GET3D on latents
        c = torch.ones(1, device='cuda')
        output = eval_get3d_single(g_ema, geo_z_edited, tex_z_edited, c)

        cur_output = output.detach()

        # Get CLIP Loss
        loss = loss_fn(output[0]) # + lmbda_1 * ((geo_z_edited - geo_z) ** 2).sum() + lmbda_2 * ((tex_z_edited - tex_z) ** 2).sum()
        
        # Backpropagation
        loss.backward()

        res_loss.append(loss[0].item())

        optimizer_geo.step()
        optimizer_geo.zero_grad()

        optimizer_tex.step()
        optimizer_tex.zero_grad()

        edited_latents.append((geo_z_edited.detach().cpu(), tex_z_edited.detach().cpu()))
    
    return original_latents, edited_latents, res_loss

if __name__ == "__main__":
    c = None
    with open('test.pickle', 'rb') as f:
        c = pickle.load(f)

    G_ema = constructGenerator(**c)

    torch.manual_seed(58)

    z = torch.randn([1, 512], device='cuda')  # random code for geometry
    tex_z = torch.randn([1, 512], device='cuda')  # random code for texture

    original, edited, loss = train_eval(G_ema, z, tex_z, 'Ferrari', 100)

    print(loss)

    result = []

    with torch.no_grad():
        G_ema.eval()
        img_original = eval_get3d_single(G_ema, original[0].to('cuda'), original[1].to('cuda'), torch.ones(1, device='cuda'))
        img_edited = eval_get3d_single(G_ema, edited[-1][0].to('cuda'), edited[-1][1].to('cuda'), torch.ones(1, device='cuda'))
        result.append((img_original.cpu(), img_edited.cpu()))
    with open('output_img.pickle', 'wb') as f:
        pickle.dump(result, f)
    
