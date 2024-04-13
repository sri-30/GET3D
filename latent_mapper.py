import torch
import numpy as np
import copy
import pickle
import time

from clip_utils.clip_loss_nada import CLIPLoss
from training_utils.training_utils import get_lr
from get3d_utils import constructGenerator, save_textured_mesh, eval_get3d_angles, generate_random_camera, generate_rotate_camera_list
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('./runs/')

class LayerTransform(torch.nn.Module):
    def __init__(self):
        super(LayerTransform, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class LatentMapper(torch.nn.Module):
    def __init__(self, freeze_geo=False, freeze_tex=False):
        super(LatentMapper, self).__init__()
        self.num_ws_geo = 22
        self.num_ws_tex = 9

        # self.geo_mapper = torch.nn.ModuleList([LayerTransform() for _ in range(self.num_ws_geo)])
        # self.tex_mapper = torch.nn.ModuleList([LayerTransform() for _ in range(self.num_ws_tex)])
        
        
        # self.tex_mapper = LayerTransform()
        
        #self.geo_mapper_nums = [0, 5, 11, 15, 20, 22]
        self.geo_mapper_nums = [0, 22]
        self.geo_mapper = torch.nn.ModuleList([LayerTransform() for _ in self.geo_mapper_nums[1:]])

        # self.geo_mapper_triplane = torch.nn.ModuleList([LayerTransform() for _ in self.geo_mapper_nums[1:]])
        # self.geo_mapper_sdf_def = LayerTransform()
        # self.geo_mapper = LayerTransform()

        # self.tex_mapper = LayerTransform()

        #self.apply(self._init_weights)

        # ws_geo
        # 0 -> tri_plane_synthesis_num_ws_geo : features
        # 0 -> 1, 2->4,  5->7, 8->10, 11->13, 15->16, 17->19
        # 20 -> 22 : sdf, def

        # ws_tex
        # 0 -> tri_plane_synthesis_num_ws_tex : features
        # 0 -> 0, 1->1, 2->2, 3->3, 4->4, 5->5 6->tri_plane_synthesis_num_tex_geo - 1
        # 9 -> 11 : rgb
    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
    

    def forward(self, ws_geo, ws_tex):
        add_geo = torch.cat([self.geo_mapper[i](ws_geo[:,self.geo_mapper_nums[i]:num]) for i, num in enumerate(self.geo_mapper_nums[1:])], dim=1)
        # print([(i, num) for i, num in enumerate(self.geo_mapper_nums[1:])])
        # print([self.geo_mapper[i](ws_geo[:,self.geo_mapper_nums[i]:num]) for i, num in enumerate(self.geo_mapper_nums[1:])])
        # print(add_geo.shape)
        # add_tex = self.tex_mapper(ws_tex)
        return ws_geo + add_geo, ws_tex
        #return ws_geo + torch.cat([self.geo_mapper[i](ws_geo[:,self.geo_mapper_nums[i-1]:num]) for i, num in enumerate(self.geo_mapper_nums[1:])], dim=1), ws_tex + self.tex_mapper(ws_tex)

def train_eval(G, text_prompt, n_epochs=5, lmbda_1=0.0015, lmbda_2=0.0015, intermediate_space=False, loss_type = 'global', corpus_type='body_type'):
    camera_list = generate_rotate_camera_list()
    camera_list = [camera_list[4]]
    
    g_ema = copy.deepcopy(G).eval()

    if corpus_type == 'body_type':
        corpus = ['Sports Car', 'SUV', 'Hatchback', 'Sedan']
    elif corpus_type == 'textures':
        corpus = ["Paint", "Glass", "Chrome", "Rubber", "Leather", "Plastic", "Carbon Fiber", "Fabric", "Metal", "Rust", "Dirt", "Decal", "Reflective Surfaces", "Ice", "Wood", "Neon", "Matte Black", "Carbon Ceramic"]
    else:
        raise NotImplementedError


    clip_loss = CLIPLoss(source_text='car', target_text=text_prompt, corpus=['Sports Car', 'SUV', 'Hatchback', 'Sedan'], aux_string='')

    g_ema = copy.deepcopy(G).eval()

    latent_mapper = LatentMapper().to('cuda')
    latent_mapper.train()
    latent_mapper.requires_grad_(True)

    learning_rate = 1e-4
    z_optim = torch.optim.Adam(latent_mapper.parameters(), lr=learning_rate)

    n_batch = 3

    n_training_samples = 90
    train_z_geo = torch.randn(n_training_samples, G.z_dim, device='cuda')
    train_z_tex = torch.randn(n_training_samples, G.z_dim, device='cuda')

    n_val = 3
    validation_geo = torch.randn(n_val, G.z_dim, device='cuda')
    validation_tex = torch.randn(n_val, G.z_dim, device='cuda')

    z_geo_split = torch.split(train_z_geo, n_batch, dim=0)
    z_tex_split = torch.split(train_z_tex, n_batch, dim=0)

    metrics_save = {'Training CLIP Loss': [], 'Validation CLIP Loss': [], 'Geo Loss': [], 'Tex Loss': []}

    # writer.add_graph(latent_mapper, (validation_geo[0].unsqueeze(0), validation_tex[0].unsqueeze(0)))
    # writer.close()

    # init_optim = torch.optim.SGD(latent_mapper.parameters(), lr=1e-3)

    # for _ in range(100):
    #     with torch.no_grad():
    #             z_geo = torch.randn(5, G.z_dim, device='cuda')
    #             z_tex = torch.randn(5, G.z_dim, device='cuda')
    #             ws_geo = g_ema.mapping_geo(z_geo, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)
    #             ws_tex = g_ema.mapping(z_tex, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)
    #     ws_geo_edited, ws_tex_edited = latent_mapper(ws_geo, ws_tex)
    #     loss = ((ws_geo_edited - ws_geo) ** 2).sum() + ((ws_tex_edited - ws_tex) ** 2).sum()
    #     loss.backward()
    #     init_optim.step()
    #     init_optim.zero_grad()

    for i in range(n_epochs):
        print(f'Epoch: {i}')
        
        for j in range(len(z_geo_split)):
            print(f'Batch: {j}')
            loss_log = 0
            loss_geo_log = 0
            loss_tex_log = 0
            for k in range(len(z_geo_split[j])):
                z_geo = z_geo_split[j][k].unsqueeze(0)
                z_tex = z_tex_split[j][k].unsqueeze(0)

                with torch.no_grad():
                    cameras = generate_random_camera(1, n_views=7)
                    cameras = cameras.unsqueeze(0)
                    cameras = cameras.transpose(0, 2)

                    ws_geo = g_ema.mapping_geo(z_geo, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)
                    ws_tex = g_ema.mapping(z_tex, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)

                    output_original = eval_get3d_angles(g_ema,  ws_geo, ws_tex, cameras=cameras,  intermediate_space=True)

                # ws_geo.requires_grad_(True)
                # ws_tex.requires_grad_(True)
                ws_geo_edited, ws_tex_edited = latent_mapper(ws_geo, ws_tex)

                output_edited = eval_get3d_angles(g_ema, ws_geo_edited, ws_tex_edited,
                                            cameras=cameras, intermediate_space=intermediate_space)
            
                # Get CLIP Loss
                if loss_type == 'global':
                    loss_clip = clip_loss.global_loss(output_edited)
                elif loss_type == 'pae':
                    loss_clip = clip_loss.projection_augmentation_loss_nada(output_original, output_edited, power=8)
                elif loss_type == 'directional':
                    loss_clip = clip_loss.directional_loss(output_original, output_edited).mean()
                else:
                    raise NotImplementedError

                # Control similarity to original latents
                loss_geo = lmbda_1 * ((ws_geo_edited - ws_geo) ** 2).mean(dim=1).sum()
                loss_tex = lmbda_2 * ((ws_tex_edited - ws_tex) ** 2).mean(dim=1).sum()

                loss = loss_clip + loss_geo + loss_tex
                loss.backward()

                with torch.no_grad():
                    loss_log += loss_clip.item()
                    loss_geo_log += loss_geo.item()
                    loss_tex_log += loss_tex.item()
            
            # for param in latent_mapper.parameters():
            #     print(param.grad.norm())

            # for param in latent_mapper.parameters():
            #     if not (param.grad is None):
            #         # print(param.grad.norm())
            #         torch.nn.utils.clip_grad_norm_(param.grad.norm(), 0.05)

            torch.nn.utils.clip_grad_norm_(latent_mapper.parameters(), 0.1)

            # Update Optimizer
            z_optim.step()
            z_optim.zero_grad()
            torch.cuda.empty_cache()
            # print(loss_geo_log/len(z_geo_split[j]))
            with torch.no_grad():
                
                val_ws_geo = g_ema.mapping_geo(validation_geo, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)
                val_ws_tex = g_ema.mapping(validation_tex, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)
                
                latent_mapper.eval()
                latent_mapper.requires_grad_(False)

                val_ws_geo_edited, val_ws_tex_edited = latent_mapper(val_ws_geo, val_ws_tex)

                latent_mapper.train()
                latent_mapper.requires_grad_(True)
                
                validation_img_original = eval_get3d_angles(g_ema, val_ws_geo, val_ws_tex, cameras=camera_list, intermediate_space=True)

                validation_img_edited = eval_get3d_angles(g_ema, val_ws_geo_edited, val_ws_tex_edited, cameras=camera_list, intermediate_space=True)
                if loss_type == 'global':
                    loss_ = clip_loss.global_loss(validation_img_edited)
                elif loss_type == 'pae':
                    loss_ = clip_loss.projection_augmentation_loss_nada(validation_img_original, validation_img_edited)
                elif loss_type == 'directional':
                    loss_ = clip_loss(validation_img_original, validation_img_edited).sum()
                else:
                    raise NotImplementedError

                metrics_save['Training CLIP Loss'].append(loss_log / len(z_geo_split[j]))
                metrics_save['Validation CLIP Loss'].append(loss_.item())
                
                metrics_save['Geo Loss'].append(loss_geo_log / len(z_geo_split[j]))
                metrics_save['Tex Loss'].append(loss_tex_log / len(z_geo_split[j]))

    return latent_mapper, metrics_save

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
    n_epochs = 10
    intermediate_space=True

    torch.manual_seed(random_seed)

    mapper, metrics_save = train_eval(G_ema, text_prompt, n_epochs=n_epochs, lmbda_1=lmbda_1, lmbda_2=lmbda_2, intermediate_space=intermediate_space, loss_type=loss_type_)

    torch.save(mapper, f'latent_mapper_saved/{text_prompt}_{loss_type_}_mapper.pt')

    with open(f'latent_mapper_{text_prompt}_{loss_type_}_metrics.pickle', 'wb') as f:
        pickle.dump(metrics_save, f)

