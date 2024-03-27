import torch
import dnnlib
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import copy

import numpy as np

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

    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(True).to(
        device)  # subclass of torch.nn.Module
    # D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(
    #     device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    if resume_pretrain is not None and (rank == 0):
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G.load_state_dict(model_state_dict['G'], strict=True)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        # D.load_state_dict(model_state_dict['D'], strict=True)
    return G_ema

def eval_get3d(G_ema, grid_z, grid_tex_z, grid_c):
    G_ema.update_w_avg()
    camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=grid_z[0].shape[0])
    if grid_tex_z is None:
        grid_tex_z = grid_z
    for i_camera, camera in enumerate(camera_list):
        images_list = []
        for z, geo_z, c in zip(grid_tex_z, grid_z, grid_c):
            print(z)
            img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
                z=z, geo_z=geo_z, c=c, noise_mode='const',
                generate_no_light=True, truncation_psi=0.7, camera=camera)
            rgb_img = img[:, :3]
            save_img = rgb_img.detach()
            images_list.append(save_img.cpu().numpy())
    images = np.concatenate(images_list, axis=0)
    return images

def eval_get3d_tensor(G_ema, grid_z, grid_tex_z, grid_c):
    G_ema.update_w_avg()
    camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=grid_z[0].shape[0])
    camera = camera_list[4]
    output_tensor = None
    for i, _ in enumerate(grid_z):
        geo_z = grid_z[i]
        tex_z = grid_tex_z[i]
        img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
            z=tex_z, geo_z=geo_z, c=grid_c, noise_mode='const',
            generate_no_light=True, truncation_psi=0.7, camera=camera)
        rgb_img = img[:, :3]
        if output_tensor is None:
            output_tensor = torch.cat((rgb_img, ))
        else:
            output_tensor = torch.cat((output_tensor, rgb_img))
    return output_tensor

def eval_get3d_single(G_ema, geo_z, tex_z, grid_c):
    camera_list = G_ema.synthesis.generate_rotate_camera_list()
    camera = camera_list[4]
    img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
        z=tex_z, geo_z=geo_z, c=grid_c, noise_mode='const',
        generate_no_light=True, truncation_psi=0.7, camera=camera)
    rgb_img = img[:, :3]
    return rgb_img


def eval_get3d_weights_ws(G_ema, ws_geo, ws, grid_c):
    with torch.no_grad():
        camera_list = G_ema.synthesis.generate_rotate_camera_list()
        camera = camera_list[4]
    img, syn_camera, mask_pyramid, sdf_reg_loss, render_return_value = G_ema.synthesis(
        ws, return_shape=False,
        ws_geo=ws_geo, camera=camera)
    return img[:, :3]

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

def freeze_generator_layers(g_ema):
    g_ema.synthesis.requires_grad_(False)

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

def determine_opt_layers(g_frozen, g_train, clip_loss, auto_layer_batch=3, auto_layer_iters=1, auto_layer_k=20):
    """
    original code : return chosen layers : List[nn.Modules, nn.Modules, ...]
    this code     : return chosen layers idx : List[int, int, ...], List[int, int, ...]
                    * note that this returns two list for tex. and geo.
    """
    z_dim = 512
    c_dim = 0
    sample_z_tex = torch.randn(auto_layer_batch, z_dim, device='cuda')
    sample_z_geo = torch.randn(auto_layer_batch, z_dim, device='cuda')

    with torch.no_grad():
        initial_w_tex_codes = g_frozen.mapping(sample_z_tex, c_dim)  # (B, 9, 512)
        initial_w_geo_codes = g_frozen.mapping_geo(sample_z_geo, c_dim)  # (B, 22, 512)

    w_tex_codes = torch.Tensor(initial_w_tex_codes.cpu().detach().numpy()).to('cuda')
    w_geo_codes = torch.Tensor(initial_w_geo_codes.cpu().detach().numpy()).to('cuda')

    w_tex_codes.requires_grad = True
    w_geo_codes.requires_grad = True

    w_optim = torch.optim.SGD([w_tex_codes, w_geo_codes], lr=0.01)

    for _ in range(auto_layer_iters):
        generated_from_w, _ = generate_img_layer(g_train, ws=w_tex_codes, ws_geo=w_geo_codes)  # (B, C, H, W)
        w_loss = clip_loss.global_loss(generated_from_w).sum()

        w_optim.zero_grad()
        w_loss.backward()
        w_optim.step()

    layer_tex_weights = torch.abs(w_tex_codes - initial_w_tex_codes).mean(dim=-1).mean(dim=0)
    layer_geo_weights = torch.abs(w_geo_codes - initial_w_geo_codes).mean(dim=-1).mean(dim=0)

    cutoff = len(layer_tex_weights)

    chosen_layers_idx = torch.topk(torch.cat([layer_tex_weights, layer_geo_weights], dim=0), auto_layer_k)[
        1].cpu().numpy().tolist()
    chosen_layer_idx_tex = []
    chosen_layer_idx_geo = []
    for idx in chosen_layers_idx:
        if idx >= cutoff:
            chosen_layer_idx_geo.append(idx - cutoff)
        else:
            chosen_layer_idx_tex.append(idx)

    return chosen_layer_idx_tex, chosen_layer_idx_geo

def generate_img_layer(
    G_ema,
    ws,
    ws_geo,
    texture_resolution=2048
):
    syn = G_ema.synthesis
    # (1) Generate 3D mesh first
    # NOTE :
    # this code is shared by 'def generate' and 'def extract_3d_mesh'
    if syn.one_3d_generator:
        sdf_feature, tex_feature = syn.generator.get_feature(
            ws[:, :syn.generator.tri_plane_synthesis.num_ws_tex],
            ws_geo[:, :syn.generator.tri_plane_synthesis.num_ws_geo])
        ws = ws[:, syn.generator.tri_plane_synthesis.num_ws_tex:]
        ws_geo = ws_geo[:, syn.generator.tri_plane_synthesis.num_ws_geo:]
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = syn.get_geometry_prediction(ws_geo, sdf_feature)
    else:
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = syn.get_geometry_prediction(ws_geo)

    ws_tex = ws

    # (2) Generate random camera
    with torch.no_grad():
        campos, cam_mv, rotation_angle, elevation_angle, sample_r = syn.generate_random_camera(
            ws_tex.shape[0], n_views=1)
        gen_camera = (campos, cam_mv, sample_r, rotation_angle, elevation_angle)
        run_n_view = 1

    # NOTE
    # tex_pos: Position we want to query the texture field || List[(1,1024, 1024,3) * Batch]
    # tex_hard_mask = 2D silhoueete of the rendered image  || Tensor(Batch, 1024, 1024, 1)

    # (3) Render the mesh into 2D image (get 3d position of each image plane)
    antilias_mask, hard_mask, return_value = syn.render_mesh(mesh_v, mesh_f, cam_mv)

    tex_pos = return_value['tex_pos']
    tex_hard_mask = hard_mask

    tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
    tex_hard_mask = torch.cat(
        [torch.cat(
            [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
                for i_view in range(run_n_view)], dim=2)
            for i in range(ws_tex.shape[0])], dim=0)

    # (4) Querying the texture field to predict the texture feature for each pixel on the image
    if syn.one_3d_generator:
        tex_feat = syn.get_texture_prediction(ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask, tex_feature)
    else:
        tex_feat = syn.get_texture_prediction(
            ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask)
    background_feature = torch.zeros_like(tex_feat)

    # (5) Merge them together
    img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

    # NOTE : debug -> no need to execute (6)
    # (6) We should split it back to the original image shape

    ws_list = [ws_tex[i].unsqueeze(dim=0).expand(return_value['tex_pos'][i].shape[0], -1, -1) for i in
               range(len(return_value['tex_pos']))]
    ws = torch.cat(ws_list, dim=0).contiguous()

    # (7) Predict the RGB color for each pixel (syn.to_rgb is 1x1 convolution)
    if syn.feat_channel > 3:
        network_out = syn.to_rgb(img_feat.permute(0, 3, 1, 2), ws[:, -1])
    else:
        network_out = img_feat.permute(0, 3, 1, 2)

    img = network_out

    img = img[:, :3]

    return img, None

def eval_get3d_angles(G_ema, z_geo, z_tex, camera_idx=[4], intermediate_space=False):
    if intermediate_space:
        ws_geo = z_geo
        ws_tex = z_tex
    else:
        ws_geo = G_ema.mapping_geo(z_geo, c=torch.ones(1, device='cuda'), truncation_psi=0.7, update_emas=False)
        ws_tex = G_ema.mapping(z_tex, c=torch.ones(1, device='cuda'), truncation_psi=0.7, update_emas=False)
    # -------------------      generate    ------------------- #

    # (1) Generate 3D mesh first
    # NOTE :
    # this code is shared by 'def generate' and 'def extract_3d_mesh'
    ws = ws_tex
    syn = G_ema.synthesis
    if syn.one_3d_generator:
        sdf_feature, tex_feature = syn.generator.get_feature(
            ws[:, :syn.generator.tri_plane_synthesis.num_ws_tex],
            ws_geo[:, :syn.generator.tri_plane_synthesis.num_ws_geo])
        ws = ws[:, syn.generator.tri_plane_synthesis.num_ws_tex:]
        ws_geo = ws_geo[:, syn.generator.tri_plane_synthesis.num_ws_geo:]
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = syn.get_geometry_prediction(ws_geo, sdf_feature)
    else:
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = syn.get_geometry_prediction(ws_geo)

    ws_tex = ws

    # (2) Generate random camera
    with torch.no_grad():
        cameras = syn.generate_rotate_camera_list()
    # NOTE
    # tex_pos: Position we want to query the texture field || List[(1,1024, 1024,3) * Batch]
    # tex_hard_mask = 2D silhoueete of the rendered image  || Tensor(Batch, 1024, 1024, 1)

    antilias_mask = []
    tex_pos = []
    tex_hard_mask = []
    return_value = {'tex_pos': []}

    for idx in camera_idx:
        cam = cameras[idx]
        antilias_mask_, hard_mask_, return_value_ = syn.render_mesh(mesh_v, mesh_f, cam)
        antilias_mask.append(antilias_mask_)
        tex_hard_mask.append(hard_mask_)

        for pos in return_value_['tex_pos']:
            return_value['tex_pos'].append(pos)

    antilias_mask = torch.cat(antilias_mask, dim=0)  # (B*n_view, 1024, 1024, 1)
    tex_hard_mask = torch.cat(tex_hard_mask, dim=0)  # (B*n_view, 1024, 1024, 3)
    tex_pos = return_value['tex_pos']

    ws_tex = ws_tex.repeat(len(camera_idx), 1, 1)
    ws_geo = ws_geo.repeat(len(camera_idx), 1, 1)
    tex_feature = tex_feature.repeat(len(camera_idx), 1, 1, 1)

    # (4) Querying the texture field to predict the texture feature for each pixel on the image
    if syn.one_3d_generator:
        tex_feat = syn.get_texture_prediction(ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask, tex_feature)
    else:
        tex_feat = syn.get_texture_prediction(
            ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask)
    background_feature = torch.zeros_like(tex_feat)

    # (5) Merge them together
    img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

    # NOTE : debug -> no need to execute (6)
    # (6) We should split it back to the original image shape

    ws_list = [ws_tex[i].unsqueeze(dim=0).expand(return_value['tex_pos'][i].shape[0], -1, -1) for i in
               range(len(return_value['tex_pos']))]
    ws = torch.cat(ws_list, dim=0).contiguous()

    # (7) Predict the RGB color for each pixel (syn.to_rgb is 1x1 convolution)
    if syn.feat_channel > 3:
        network_out = syn.to_rgb(img_feat.permute(0, 3, 1, 2), ws[:, -1])
    else:
        network_out = img_feat.permute(0, 3, 1, 2)

    img = network_out

    img = img[:, :3]

    return img

def eval_nada_z(G_ema, G_ema_frozen, z_geo, z_tex, n_cameras=2):
    ws_geo = G_ema.mapping_geo(z_geo, c=0)
    ws_tex = G_ema.mapping(z_tex, c=0)
    return eval_nada(G_ema, G_ema_frozen, ws_tex, ws_geo, n_cameras=n_cameras)

def eval_nada(
        G_ema,
        G_ema_frozen,
        ws,
        ws_geo,
        texture_resolution=2048,
        n_cameras=2
):

    # -------------------      generate    ------------------- #

    # (1) Generate 3D mesh first
    # NOTE :
    # this code is shared by 'def generate' and 'def extract_3d_mesh'
    syn = G_ema.synthesis
    if syn.one_3d_generator:
        sdf_feature, tex_feature = syn.generator.get_feature(
            ws[:, :syn.generator.tri_plane_synthesis.num_ws_tex],
            ws_geo[:, :syn.generator.tri_plane_synthesis.num_ws_geo])
        ws = ws[:, syn.generator.tri_plane_synthesis.num_ws_tex:]
        ws_geo = ws_geo[:, syn.generator.tri_plane_synthesis.num_ws_geo:]
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = syn.get_geometry_prediction(ws_geo, sdf_feature)
    else:
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = syn.get_geometry_prediction(ws_geo)

    ws_tex = ws

    # (2) Generate random camera
    with torch.no_grad():
        campos, cam_mv, rotation_angle, elevation_angle, sample_r = G_ema_frozen.synthesis.generate_random_camera(
            ws_tex.shape[0], n_views=n_cameras)
        gen_camera = (campos, cam_mv, sample_r, rotation_angle, elevation_angle)
        run_n_view = n_cameras

    # NOTE
    # tex_pos: Position we want to query the texture field || List[(1,1024, 1024,3) * Batch]
    # tex_hard_mask = 2D silhoueete of the rendered image  || Tensor(Batch, 1024, 1024, 1)

    antilias_mask = []
    tex_pos = []
    tex_hard_mask = []
    return_value = {'tex_pos': []}

    for idx in range(n_cameras):
        cam = cam_mv[:, idx, :, :].unsqueeze(1)
        antilias_mask_, hard_mask_, return_value_ = syn.render_mesh(mesh_v, mesh_f, cam)
        antilias_mask.append(antilias_mask_)
        tex_hard_mask.append(hard_mask_)

        for pos in return_value_['tex_pos']:
            return_value['tex_pos'].append(pos)

    antilias_mask = torch.cat(antilias_mask, dim=0)  # (B*n_view, 1024, 1024, 1)
    tex_hard_mask = torch.cat(tex_hard_mask, dim=0)  # (B*n_view, 1024, 1024, 3)
    tex_pos = return_value['tex_pos']

    ws_tex = ws_tex.repeat(n_cameras, 1, 1)
    ws_geo = ws_geo.repeat(n_cameras, 1, 1)
    tex_feature = tex_feature.repeat(n_cameras, 1, 1, 1)

    # (4) Querying the texture field to predict the texture feature for each pixel on the image
    if syn.one_3d_generator:
        tex_feat = syn.get_texture_prediction(ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask, tex_feature)
    else:
        tex_feat = syn.get_texture_prediction(
            ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask)
    background_feature = torch.zeros_like(tex_feat)

    # (5) Merge them together
    img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

    # NOTE : debug -> no need to execute (6)
    # (6) We should split it back to the original image shape

    ws_list = [ws_tex[i].unsqueeze(dim=0).expand(return_value['tex_pos'][i].shape[0], -1, -1) for i in
               range(len(return_value['tex_pos']))]
    ws = torch.cat(ws_list, dim=0).contiguous()

    # (7) Predict the RGB color for each pixel (syn.to_rgb is 1x1 convolution)
    if syn.feat_channel > 3:
        network_out = syn.to_rgb(img_feat.permute(0, 3, 1, 2), ws[:, -1])
    else:
        network_out = img_feat.permute(0, 3, 1, 2)

    img = network_out

    img = img[:, :3]

    return_generate = [img, antilias_mask]
    return return_generate[0]

def eval_get3d_weights(G_ema, geo_z, tex_z, grid_c):
    # Step 1: Map the sampled z code to w-space
    ws = G_ema.mapping(tex_z, grid_c, update_emas=False)
    # geo_z = torch.randn_like(z)
    ws_geo = G_ema.mapping_geo(
        geo_z, grid_c,
        update_emas=False)

    # # # Step 2: Apply style mixing to the latent code
    # if self.style_mixing_prob > 0:
    #     cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
    #     cutoff = torch.where(
    #         torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
    #         torch.full_like(cutoff, ws.shape[1]))
    #     ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]

    #     cutoff = torch.empty([], dtype=torch.int64, device=ws_geo.device).random_(1, ws_geo.shape[1])
    #     cutoff = torch.where(
    #         torch.rand([], device=ws_geo.device) < self.style_mixing_prob, cutoff,
    #         torch.full_like(cutoff, ws_geo.shape[1]))
    #     ws_geo[:, cutoff:] = self.G.mapping_geo(torch.randn_like(z), c, update_emas=False)[:, cutoff:]

    # Step 3: Generate rendered image of 3D generated shapes.
    with torch.no_grad():
        camera_list = G_ema.synthesis.generate_rotate_camera_list()
        camera = camera_list[4]
    img, syn_camera, mask_pyramid, sdf_reg_loss, render_return_value = G_ema.synthesis(
        ws, return_shape=False,
        ws_geo=ws_geo, camera=camera)
    return img[:, :3]

def eval_get3d_single_intermediates(G_ema, ws_geo, ws_tex, grid_c):
    camera_list = G_ema.synthesis.generate_rotate_camera_list()
    camera = camera_list[4]
    img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d_intermediate(
        ws_geo=ws_geo, ws=ws_tex, c=grid_c, noise_mode='const',
        generate_no_light=True, truncation_psi=0.7, camera=camera)
    rgb_img = img[:, :3]
    return rgb_img

def intermediates(G_ema, geo_z, tex_z, grid_c):
    G_ema.update_w_avg()
    return G_ema.get_intermediates(geo_z, tex_z, grid_c)