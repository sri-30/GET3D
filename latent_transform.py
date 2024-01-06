import torch
import numpy as np
import copy
import pickle

from clip_utils.clip_loss import CLIPLoss
from get3d_utils import constructGenerator, eval_get3d_single, intermediates, eval_get3d_single_intermediates

class TransformIntermediateLatent(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(512*n, 1024*n),
            torch.nn.ReLU(),
            torch.nn.Linear(1024*n, 512*n),
            torch.nn.ReLU(),
            torch.nn.Linear(512*n, 512*n),
        )

    def forward(self, x):
        x_flat = self.flatten(x)
        logits = self.linear_relu_stack(x_flat)
        return logits.reshape(x.shape)

def preprocess_rgb(array):
    lo, hi = -1, 1
    img = array
    img = img.transpose(1, 3)
    img = img.transpose(1, 2)
    img = (img - lo) * (255 / (hi - lo))
    img.clip(0, 255)
    return img

def train_eval(G, data_geo_ws, data_tex_ws, text_prompt, n_epochs=5, lmbda_1=0.0015, lmbda_2=0.0015):
    n_geo = 22
    n_tex = 9

    model_geo = TransformIntermediateLatent(n_geo).to('cuda')
    model_tex = TransformIntermediateLatent(n_tex).to('cuda')

    model_geo.train()
    model_tex.train()

    learning_rate_geo = 1e-3
    optimizer_geo = torch.optim.SGD(model_geo.parameters(), lr=learning_rate_geo)

    learning_rate_tex = 1e-3
    optimizer_tex = torch.optim.SGD(model_tex.parameters(), lr=learning_rate_tex)
    
    loss_fn = CLIPLoss(text_prompt)

    original_latents = (data_geo_ws.detach().cpu(), data_tex_ws.detach().cpu())
    edited_latents = []
    res_loss = []

    min_latent = None
    min_loss = float('inf')

    for i in range(n_epochs):
        print(i)

        geo_ws = data_geo_ws.to('cuda').detach()
        tex_ws = data_tex_ws.to('cuda').detach()

        geo_ws.requires_grad = True
        tex_ws.requires_grad = True

        g_ema = copy.deepcopy(G).eval()

        # # Transform latents with model
        geo_ws_edited = model_geo(geo_ws)
        tex_ws_edited = model_tex(tex_ws)

        # Get output of GET3D on latents
        #output = eval_get3d_single(g_ema, torch.randn([1, 512], device='cuda'), torch.randn([1, 512], device='cuda'), torch.ones(1, device='cuda'))
        output = eval_get3d_single_intermediates(g_ema, geo_ws_edited, tex_ws_edited, torch.ones(1, device='cuda'))

        # Get CLIP Loss
        loss = loss_fn(output[0]) + lmbda_1 * ((geo_ws_edited - geo_ws) ** 2).sum() + lmbda_2 * ((tex_ws_edited - tex_ws) ** 2).sum()
        # Backpropagation
        loss.backward()

        if loss[0].item() < min_loss:
            min_loss = loss[0].item()
            min_latent = (geo_ws_edited.detach().cpu(), tex_ws_edited.detach().cpu())

        res_loss.append(loss[0].item())

        optimizer_geo.step()
        optimizer_geo.zero_grad()

        optimizer_tex.step()
        optimizer_tex.zero_grad()

        edited_latents.append((geo_ws_edited.detach().cpu(), tex_ws_edited.detach().cpu()))
    
    return original_latents, edited_latents, res_loss, min_latent

if __name__ == "__main__":
    n_geo = 22
    n_tex = 9

    c = None
    with open('test.pickle', 'rb') as f:
        c = pickle.load(f)

    G = constructGenerator(**c)

    torch.manual_seed(53)

    with open('intermediates.pickle', 'rb') as f:
        intermediates_cpu = pickle.load(f)

    data_ws_geo = intermediates_cpu[4][0].to('cuda')  # random code for geometry
    data_ws_tex = intermediates_cpu[4][1].to('cuda')  # random code for texture

    original, edited, loss, min_latent = train_eval(G, data_ws_geo, data_ws_tex, 'Sports Car', 3000, lmbda_1=0.0005, lmbda_2=0.0005)

    print(loss)

    result = []

    with torch.no_grad():
        G.eval()
        img_original = eval_get3d_single_intermediates(G, original[0].to('cuda'), original[1].to('cuda'), torch.ones(1, device='cuda'))
        img_edited = eval_get3d_single_intermediates(G, min_latent[0].to('cuda'), min_latent[1].to('cuda'), torch.ones(1, device='cuda'))
        result.append((img_original.cpu(), img_edited.cpu()))
    with open('output_img_intermediate.pickle', 'wb') as f:
        pickle.dump(result, f)
    
