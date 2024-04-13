import torch
import numpy as np
import copy
import pickle
import time

from clip_utils.clip_loss_nada import CLIPLoss
from training_utils.training_utils import get_lr
from get3d_utils import constructGenerator, eval_get3d_single, eval_get3d_angles, eval_get3d_weights_ws, eval_nada, determine_opt_layers, unfreeze_generator_layers, freeze_generator_layers, generate_random_camera, generate_rotate_camera_list
from torch_utils import misc
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(log_dir='./runs/')

def preprocess_rgb(array):
    lo, hi = -1, 1
    img = array
    img = img.transpose(1, 3)
    img = img.transpose(1, 2)
    img = (img - lo) * (255 / (hi - lo))
    img.clip(0, 255)
    return img

# grads = []

def train_eval(G, text_prompt, n_epochs=5, loss_type='global'):

    camera_list = generate_rotate_camera_list()[::4]

    g_ema_frozen = copy.deepcopy(G)
    g_ema_train = copy.deepcopy(G)
    
    g_ema_frozen.update_w_avg()
    g_ema_train.update_w_avg()
    
    g_ema_frozen.requires_grad_(False)
    g_ema_frozen.eval()

    g_ema_train.mapping.requires_grad_(False)
    g_ema_train.mapping_geo.requires_grad_(False)
    g_ema_train.train()

    learning_rate = 8e-4
    opt_params = g_ema_train.synthesis.generator.tri_plane_synthesis.parameters()
    optimizer = torch.optim.Adam(opt_params, 
                                lr=learning_rate,
                                betas=(0.9, 0.99))

    clip_loss = CLIPLoss(source_text='car', target_text=text_prompt, corpus=['Sports Car', 'SUV', 'Hatchback', 'Sedan'], aux_string='')

    n_batch = 5

    n_training_samples = 90
    train_z_geo = torch.randn(n_training_samples, G.z_dim, device='cuda')
    train_z_tex = torch.randn(n_training_samples, G.z_dim, device='cuda')

    n_val = 3
    validation_geo = torch.randn(n_val, G.z_dim, device='cuda')
    validation_tex = torch.randn(n_val, G.z_dim, device='cuda')

    with torch.no_grad():
        validation_geo = g_ema_frozen.mapping_geo(validation_geo, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)[:,0,]
        validation_tex = g_ema_frozen.mapping(validation_tex, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)[:,0,]

        train_ws_geo = g_ema_frozen.mapping_geo(train_z_geo, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)[:,0,]
        train_ws_tex = g_ema_frozen.mapping(train_z_tex, torch.ones(1, device='cuda'), update_emas=False, truncation_psi=0.7)[:,0,]

    ws_geo_split = torch.split(train_ws_geo, n_batch, dim=0)
    ws_tex_split = torch.split(train_ws_tex, n_batch, dim=0)

    training_loss = []
    validation_loss = []

    for i in range(n_epochs):
        print(f'Epoch: {i}')

        for j in range(len(ws_geo_split)):
            print(f'Batch: {j}')

            # Layer freezing
            unfreeze_generator_layers(g_ema_train, [], [])
            topk_idx_tex, topk_idx_geo = determine_opt_layers(g_ema_frozen, g_ema_train, clip_loss)
            freeze_generator_layers(g_ema_train)
            unfreeze_generator_layers(g_ema_train, topk_idx_tex, topk_idx_geo)

            loss_log = 0

            for k in range(len(ws_geo_split[j])):

                g_ema_train.train()

                ws_geo = ws_geo_split[j][k].unsqueeze(0).repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1)
                ws_tex = ws_tex_split[j][k].unsqueeze(0).repeat(1, g_ema_frozen.synthesis.generator.num_ws_tex, 1)

                # Get output of GET3D on latents
                with torch.no_grad():
                    cameras = generate_random_camera(1, n_views=7)
                    cameras = cameras.unsqueeze(0)
                    cameras = cameras.transpose(0, 2)

                    frozen_img = eval_get3d_angles(g_ema_frozen, ws_geo, ws_tex, cameras=cameras, intermediate_space=True)
                
                trainable_img = eval_get3d_angles(g_ema_train, ws_geo, ws_tex, cameras=cameras, intermediate_space=True)

                if loss_type == 'global':
                    loss = clip_loss.global_loss(trainable_img)
                elif loss_type == 'pae':
                    loss = clip_loss.projection_augmentation_loss_nada(frozen_img, trainable_img)
                elif loss_type == 'directional':
                    loss = clip_loss(frozen_img, trainable_img).sum()
                else:
                    raise NotImplementedError
                
                loss.backward()
                with torch.no_grad():
                    loss_log += loss.item()

            # for param in opt_params:
            #     print(param.grad.norm())
            # with torch.no_grad():
            #     grads.append([param.grad.norm().item() for param in g_ema_train.synthesis.parameters() if param.requires_grad])
            # for param in g_ema_train.synthesis.parameters():
            #     print(param.grad.norm())
            
            # writer.add_scalar(f"Weight Transform {text_prompt} {loss_type} 9", loss_log/len(z_geo_split[j]), i*len(z_geo_split) + j)

            # torch.nn.utils.clip_grad_norm_(g_ema_train.synthesis.generator.tri_plane_synthesis.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # Validation
            with torch.no_grad():
                training_loss.append(loss_log / n_batch)
                g_ema_train.eval()
                validation_img_original = eval_get3d_angles(g_ema_frozen, validation_geo.repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1), validation_tex.repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1), cameras=camera_list, intermediate_space=True)
                validation_img_edited = eval_get3d_angles(g_ema_train, validation_geo.repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1), validation_tex.repeat(1, g_ema_frozen.synthesis.generator.num_ws_geo, 1), cameras=camera_list, intermediate_space=True)
                if loss_type == 'global':
                    loss_ = clip_loss.global_loss(validation_img_edited)
                elif loss_type == 'pae':
                    loss_ = clip_loss.projection_augmentation_loss_nada(validation_img_original, validation_img_edited)
                elif loss_type == 'directional':
                    loss_ = clip_loss(validation_img_original, validation_img_edited).sum()
                else:
                    raise NotImplementedError
                
                # loss_ = clip_loss(img_original, img_save).sum()
                # loss_ = clip_loss.projection_augmentation_loss_nada(validation_img_original, validation_img_edited)
                validation_loss.append(loss_.item())
                # edited_images.append(img_save.cpu())
            

    return g_ema_train, training_loss, validation_loss

if __name__ == "__main__":
    import sys

    print(sys.argv)
    _, random_seed_, text_prompt_, loss_type_ = sys.argv

    with open('test.pickle', 'rb') as f:
            c = pickle.load(f)
    print(c)
    G_ema = constructGenerator(**c)

    # Parameters
    random_seed = random_seed_
    text_prompt= text_prompt_
    n_epochs = 10

    torch.manual_seed(random_seed)

    z = torch.randn([1, 512], device='cuda')  # random code for geometry
    tex_z = torch.randn([1, 512], device='cuda')  # random code for texture

    G_new, train_loss, val_loss = train_eval(G_ema, z, tex_z, text_prompt, n_epochs=n_epochs, loss_type=loss_type_)
    
    # writer.flush()
    # writer.close()

    torch.save(G_new.state_dict(), f'generators_saved/{text_prompt}_{loss_type_}_generator.pt')

    # print(loss)
    # print(min(loss))
    
    with open(f'weight_transform_{text_prompt}_{loss_type_}_train_val_loss.pickle', 'wb') as f:
        pickle.dump({'train': train_loss, 'val': val_loss}, f)

    # with open('weight_transform_grads_2.pickle', 'wb') as f:
    #     pickle.dump(grads, f)

    result = []

    # with torch.no_grad():
    #     G_new.eval()
    #     img_original = eval_get3d_single(G_ema, z, tex_z, torch.ones(1, device='cuda')).cpu()
    #     img_edited = eval_get3d_single(G_new, z, tex_z, torch.ones(1, device='cuda')).cpu()
    #     result.append({'Original': img_original, 'Edited': img_edited, 'Loss': loss, 'Edited Images': edited_images})
    # # with open(f'weight_transform_results/output_img_{random_seed}_{lmbda_1}_{lmbda_2}_{text_prompt}_{n_epochs}_{time.time()}.pickle', 'wb') as f:
    # #     pickle.dump(result, f)
    # snapshot_data = dict(
    #             G=G_ema, G_ema=G_ema)
    # all_model_dict = {'G': snapshot_data['G'].state_dict(), 'G_ema': snapshot_data['G_ema'].state_dict()}
    # torch.save(all_model_dict, f'{text_prompt}_{loss_type_}')
    
    
