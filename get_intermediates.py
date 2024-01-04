import torch
import numpy as np
import copy
import pickle

from clip_utils.clip_loss import CLIPLoss
from get3d_utils import constructGenerator, eval_get3d_single, intermediates, eval_get3d_single_intermediates

def get_intermediates(G, data_geo_z, data_tex_z):
    latents = []
    G.eval()
    for i in range(len(data_geo_z)):
        with torch.no_grad():
            latents.append(intermediates(G, data_geo_z[i], data_tex_z[i], torch.ones(1, device='cuda')))    
    return latents

if __name__ == "__main__":
    c = None
    with open('test.pickle', 'rb') as f:
        c = pickle.load(f)

    G = constructGenerator(**c)

    torch.manual_seed(0)

    z = torch.randn([20, 1, 512], device='cuda')  # random code for geometry
    tex_z = torch.randn([20, 1, 512], device='cuda')  # random code for texture

    intermediates = get_intermediates(G, z, tex_z)

    intermediates_cpu = [(geo.detach().cpu(), tex.detach().cpu()) for geo, tex in intermediates]

    with open('intermediates.pickle', 'wb') as f:
        pickle.dump(intermediates_cpu, f)

    # with torch.no_grad():
    #     G.eval()
    #     img_original = eval_get3d_single_intermediates(G, original[0].to('cuda'), original[1].to('cuda'), torch.ones(1, device='cuda'))
    #     img_edited = eval_get3d_single_intermediates(G, edited[-1][0].to('cuda'), edited[-1][1].to('cuda'), torch.ones(1, device='cuda'))
    #     result.append((img_original.cpu(), img_edited.cpu()))
    # with open('output_img_intermediate.pickle', 'wb') as f:
    #     pickle.dump(result, f)
    
