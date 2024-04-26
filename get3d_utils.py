import torch
import dnnlib
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import copy
from training.sample_camera_distribution import sample_camera, create_camera_from_angle
from training.utils.utils_3d import savemeshtes2

import numpy as np
import math

def constructGenerator(
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        inference_vis=False,
        inference_to_generate_textured_mesh=False,
        resume_pretrain=None,
        inference_save_interpolation=False,
        inference_compute_fid=False,
        inference_generate_geo=False,
        code_z=None ,
        code_tex_z=None,
        text_prompt="",
        **dummy_kawargs
):
    print("constructing")
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import filtered_lrelu

    upfirdn2d._init()
    bias_act._init()
    filtered_lrelu._init()

    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = True  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.


    common_kwargs = dict(
        c_dim=0, img_resolution=training_set_kwargs['resolution'] if 'resolution' in training_set_kwargs else 1024, img_channels=3)
    G_kwargs['device'] = device

    G_ema= dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(True).to(
        device)
    model_state_dict = torch.load(resume_pretrain, map_location=device)
    G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
    return G_ema

def get_all_generator_layers_dict(g_ema):

    layer_idx_geo = {}
    layer_idx_tex = {}

    tri_plane_blocks = g_ema.synthesis.generator.tri_plane_synthesis.children()

    idx_geo = 0
    idx_tex = 0

    # triplane
    for block in tri_plane_blocks:
        if hasattr(block, 'conv0'):
            layer_idx_geo[idx_geo] = f'b{block.resolution}.conv0'
            idx_geo += 1
        if hasattr(block, 'conv1'):
            layer_idx_geo[idx_geo] = f'b{block.resolution}.conv1'
            idx_geo += 1
        if hasattr(block, 'togeo'):
            layer_idx_geo[idx_geo] = f'b{block.resolution}.togeo'
            idx_geo += 1
        if hasattr(block, 'totex'):
            layer_idx_tex[idx_tex] = f'b{block.resolution}.totex'
            idx_tex += 1

    # mlp_synthesis
    # note that last number = ModuleList index
    layer_idx_tex[idx_tex] = 'mlp_synthesis_tex.0'
    idx_tex += 1
    layer_idx_tex[idx_tex] = 'mlp_synthesis_tex.1'

    layer_idx_geo[idx_geo] = 'mlp_synthesis_geo.0'
    idx_geo += 1
    layer_idx_geo[idx_geo] = 'mlp_synthesis_geo.1'

    return layer_idx_tex, layer_idx_geo

def freeze_generator_layers(g_ema, layer_list=None):
    if layer_list is None:
        g_ema.synthesis.requires_grad_(False)
    else:
        for layer in layer_list:
            layer.requires_grad_(True)
            

def unfreeze_generator_layers(g_ema, topk_idx_tex: list, topk_idx_geo: list):
    """
    args
        topk_idx_tex : chosen layers - geo
        topk_idx_geo : chosen layers - tex
        layer_geo_dict , layer_tex_dict : result of get_all_generator_layers()
    """
    if not topk_idx_tex and not topk_idx_geo:
        g_ema.synthesis.generator.tri_plane_synthesis.requires_grad_(True)
        return  # all unfreeze

    layer_tex_dict, layer_geo_dict = get_all_generator_layers_dict(g_ema)

    for idx_tex in topk_idx_tex:
        if idx_tex >= 7:
            # mlp_synthesis_tex
            mlp_name, layer_idx = layer_tex_dict[idx_tex].split('.')
            layer_tex = getattr(g_ema.synthesis.generator.mlp_synthesis_tex, 'layers')[int(layer_idx)]
            layer_tex.requires_grad_(True)
            g_ema.synthesis.generator.mlp_synthesis_tex.layers[int(layer_idx)] = layer_tex

        else:
            # Texture TriPlane
            block_name, layer_name = layer_tex_dict[idx_tex].split('.')
            block = getattr(g_ema.synthesis.generator.tri_plane_synthesis, block_name)
            getattr(block, layer_name).requires_grad_(True)
            setattr(g_ema.synthesis.generator.tri_plane_synthesis, block_name, block)

    for idx_geo in topk_idx_geo:
        if idx_geo >= 20:
            # mlp_synthesis_sdf
            mlp_name, layer_idx = layer_geo_dict[idx_geo].split('.')
            layer_sdf = getattr(g_ema.synthesis.generator.mlp_synthesis_sdf, 'layers')[int(layer_idx)]
            layer_sdf.requires_grad_(True)
            g_ema.synthesis.generator.mlp_synthesis_sdf.layers[int(layer_idx)] = layer_sdf
            # mlp_synthesis_def
            layer_def = getattr(g_ema.synthesis.generator.mlp_synthesis_def, 'layers')[int(layer_idx)]
            layer_def.requires_grad_(True)
            g_ema.synthesis.generator.mlp_synthesis_def.layers[int(layer_idx)] = layer_def

        else:
            # Geometry TriPlane
            block_name, layer_name = layer_geo_dict[idx_geo].split('.')
            block = getattr(g_ema.synthesis.generator.tri_plane_synthesis, block_name)
            getattr(block, layer_name).requires_grad_(True)
            setattr(g_ema.synthesis.generator.tri_plane_synthesis, block_name, block)

def determine_opt_layers(g_frozen, g_train, clip_loss, n_batch=8, epochs=1, k=20, preset_latent=None):
    z_dim = 512
    c_dim = 0
    if preset_latent is None:
        sample_z_tex = torch.randn(n_batch, z_dim, device='cuda')
        sample_z_geo = torch.randn(n_batch, z_dim, device='cuda')
    else:
        sample_z_tex, sample_z_geo = preset_latent

    with torch.no_grad():
        ws_tex_original = g_frozen.mapping(sample_z_tex, c_dim)  # (B, 9, 512)
        ws_geo_original = g_frozen.mapping_geo(sample_z_geo, c_dim)  # (B, 22, 512)

    ws_tex = ws_tex_original.clone()
    ws_geo = ws_geo_original.clone()

    ws_tex.requires_grad = True
    ws_geo.requires_grad = True

    w_optim = torch.optim.SGD([ws_tex, ws_geo], lr=0.01)

    for _ in range(epochs):
        # generated_from_w, _ = generate_img_layer(g_train, ws=ws_tex, ws_geo=ws_geo, cam_mv=g_frozen.synthesis.generate_rotate_camera_list()[4].repeat(ws_tex.shape[0], 1, 1, 1))  # (B, C, H, W)
        img_edited = eval_get3d_angles(g_train, z_geo=ws_geo, z_tex=ws_tex, cameras=[generate_rotate_camera_list()[4]], intermediate_space=True)
        w_loss = clip_loss.global_loss(img_edited).mean()

        w_loss.backward()
        w_optim.step()
        w_optim.zero_grad()

    tex_distance = (ws_tex - ws_tex_original).abs().mean(dim=-1).mean(dim=0)
    geo_distance = (ws_geo - ws_geo_original).abs().mean(dim=-1).mean(dim=0)

    cutoff = len(tex_distance)

    chosen_layers_idx = torch.topk(torch.cat([tex_distance, geo_distance], dim=0), k)[1].tolist()

    chosen_layer_idx_tex = []
    chosen_layer_idx_geo = []
    for idx in chosen_layers_idx:
        if idx >= cutoff:
            chosen_layer_idx_geo.append(idx - cutoff)
        else:
            chosen_layer_idx_tex.append(idx)

    del ws_geo_original
    del ws_geo
    del ws_tex_original
    del ws_tex
    torch.cuda.empty_cache()

    return chosen_layer_idx_tex, chosen_layer_idx_geo

def generate_rotate_camera_list(n_batch=1):
    '''
    Generate a camera list for rotating the object.
    :param n_batch:
    :return:
    '''
    n_camera = 24
    camera_radius = 1.2  # align with what ww did in blender
    camera_r = torch.zeros(n_camera, 1, device='cuda') + camera_radius
    camera_phi = torch.zeros(n_camera, 1, device='cuda') + (90.0 - 15.0) / 90.0 * 0.5 * math.pi
    camera_theta = torch.range(0, n_camera - 1, device='cuda').unsqueeze(dim=-1) / n_camera * math.pi * 2.0
    camera_theta = -camera_theta
    world2cam_matrix, camera_origin, _, _, _ = create_camera_from_angle(
        camera_phi, camera_theta, camera_r, device='cuda')
    camera_list = [world2cam_matrix[i:i + 1].expand(n_batch, -1, -1).unsqueeze(dim=1) for i in range(n_camera)]
    return camera_list

def generate_random_camera(batch_size, n_views=2):
    '''
    Sample a random camera from the camera distribution during training
    :param batch_size: batch size for the generator
    :param n_views: number of views for each shape within a batch
    :return:
    '''
    world2cam_matrix, _, _, _, _ = sample_camera(
        'shapenet_car', batch_size * n_views, 'cuda')
    mv_batch = world2cam_matrix
    return mv_batch.reshape(batch_size, n_views, 4, 4)

def save_textured_mesh(G_ema, ws_geo, ws_tex, filename='default'):
    import PIL
    import cv2
    mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map = G_ema.synthesis.extract_3d_shape(ws_tex, ws_geo)
    savemeshtes2(
        mesh_v[0].data.cpu().numpy(),
        all_uvs[0].data.cpu().numpy(),
        mesh_f[0].data.cpu().numpy(),
        all_mesh_tex_idx[0].data.cpu().numpy(),
        filename
    )
    lo, hi = (-1, 1)
    img = np.asarray(tex_map[0].permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = img.clip(0, 255)
    mask = np.sum(img.astype(float), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(float)
    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    img = img * (1 - mask) + dilate_img * mask
    img = img.clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(filename.replace('.obj', '.png'))


def eval_get3d_angles(G_ema, z_geo, z_tex, cameras=[], intermediate_space=False):
    if intermediate_space:
        ws_geo = z_geo
        ws = z_tex
    else:
        ws_geo = G_ema.mapping_geo(z_geo, c=torch.ones(1, device='cuda'), truncation_psi=0.7, update_emas=False)
        ws = G_ema.mapping(z_tex, c=torch.ones(1, device='cuda'), truncation_psi=0.7, update_emas=False)

    # Generate textured mesh
    syn = G_ema.synthesis
    sdf_feature, tex_feature = syn.generator.get_feature(
        ws[:, :syn.generator.tri_plane_synthesis.num_ws_tex],
        ws_geo[:, :syn.generator.tri_plane_synthesis.num_ws_geo])
    ws = ws[:, syn.generator.tri_plane_synthesis.num_ws_tex:]
    ws_geo = ws_geo[:, syn.generator.tri_plane_synthesis.num_ws_geo:]
    mesh_v, mesh_f, _, _, _, _ = syn.get_geometry_prediction(ws_geo, sdf_feature)

    ws_tex = ws

    antilias_mask = []
    tex_pos = []
    tex_hard_mask = []
    tex_pos = []

    # Render the mesh
    for cam in cameras:
        antilias_mask_, hard_mask_, return_value_ = syn.render_mesh(mesh_v, mesh_f, cam.repeat(z_geo.shape[0], 1, 1, 1))
        antilias_mask.append(antilias_mask_)
        tex_hard_mask.append(hard_mask_)

        for pos in return_value_['tex_pos']:
            tex_pos.append(pos)

    antilias_mask = torch.cat(antilias_mask, dim=0)
    tex_hard_mask = torch.cat(tex_hard_mask, dim=0)

    ws_tex = ws_tex.repeat(len(cameras), 1, 1)
    ws_geo = ws_geo.repeat(len(cameras), 1, 1)
    tex_feature = tex_feature.repeat(len(cameras), 1, 1, 1)

    tex_feat = syn.get_texture_prediction(ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask, tex_feature)

    background_feature = torch.zeros_like(tex_feat)

    img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

    ws_list = [ws_tex[i].unsqueeze(dim=0).expand(tex_pos[i].shape[0], -1, -1) for i in
               range(len(tex_pos))]
    ws = torch.cat(ws_list, dim=0).contiguous()

    network_out = syn.to_rgb(img_feat.permute(0, 3, 1, 2), ws[:, -1])

    img = network_out

    img = img[:, :3]

    return img
