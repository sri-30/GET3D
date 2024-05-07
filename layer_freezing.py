import torch
from get3d_utils import eval_get3d_angles, generate_rotate_camera_list

def get_triplane_layers(g_ema):

    layers_geo = []
    layers_tex = []

    for block in g_ema.synthesis.generator.tri_plane_synthesis.children():
        if hasattr(block, 'conv0'):
            layers_geo.append(f'b{block.resolution}.conv0')
        if hasattr(block, 'conv1'):
            layers_geo.append(f'b{block.resolution}.conv1')
        if hasattr(block, 'togeo'):
            layers_geo.append(f'b{block.resolution}.togeo')
        if hasattr(block, 'totex'):
            layers_tex.append(f'b{block.resolution}.totex')

    return layers_geo, layers_tex

def control_layers(g_ema, layers_tex=[], layers_geo=[], status=False, opt_mlp=False):

    tri_plane_geo, tri_plane_tex = get_triplane_layers(g_ema)

    if not layers_tex and not layers_geo:
        g_ema.requires_grad_(status)

    for layer_tex in layers_tex:
        if layer_tex >= 7:
            # MLP Texture Synthesis Layers
            tex_layers = getattr(g_ema.synthesis.generator.mlp_synthesis_tex, 'layers')
            tex_layers.requires_grad_(opt_mlp and status)
        else:
            # Texture Triplane Synthesis Layers
            res, layer = tri_plane_tex[layer_tex].split('.')
            block = getattr(g_ema.synthesis.generator.tri_plane_synthesis, res)
            getattr(block, layer).requires_grad_(True)
            setattr(g_ema.synthesis.generator.tri_plane_synthesis, res, block)
    
    for layer_geo in layers_geo:
        if layer_geo >= 20:
            # MLP Geometry Synthesis Layers
            sdf_layers = getattr(g_ema.synthesis.generator.mlp_synthesis_sdf, 'layers')
            sdf_layers.requires_grad_(opt_mlp and status)
            def_layers = getattr(g_ema.synthesis.generator.mlp_synthesis_def, 'layers')
            def_layers.requires_grad_(opt_mlp and status)
        else:
            res, layer = tri_plane_geo[layer_geo].split('.')
            block = getattr(g_ema.synthesis.generator.tri_plane_synthesis, res)
            getattr(block, layer).requires_grad_(True)
            setattr(g_ema.synthesis.generator.tri_plane_synthesis, res, block)

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

    layers_tex = []
    layers_geo = []
    for idx in chosen_layers_idx:
        if idx >= cutoff:
            layers_geo.append(idx - cutoff)
        else:
            layers_tex.append(idx)

    return layers_tex, layers_geo

# def determine_opt_layers(g_frozen, g_train, clip_loss, n_batch=8, epochs=1, k=20, preset_latent=None):
#     z_dim = 512
#     c_dim = 0
#     if preset_latent is None:
#         sample_z_tex = torch.randn(n_batch, z_dim, device='cuda')
#         sample_z_geo = torch.randn(n_batch, z_dim, device='cuda')
#     else:
#         sample_z_tex, sample_z_geo = preset_latent

#     with torch.no_grad():
#         ws_tex_original = g_frozen.mapping(sample_z_tex, c_dim)  # (B, 9, 512)
#         ws_geo_original = g_frozen.mapping_geo(sample_z_geo, c_dim)  # (B, 22, 512)

#     ws_tex = ws_tex_original.clone()
#     ws_geo = ws_geo_original.clone()

#     ws_tex.requires_grad = True
#     ws_geo.requires_grad = True

#     w_optim = torch.optim.SGD([ws_tex, ws_geo], lr=0.01)

#     for _ in range(epochs):
#         # generated_from_w, _ = generate_img_layer(g_train, ws=ws_tex, ws_geo=ws_geo, cam_mv=g_frozen.synthesis.generate_rotate_camera_list()[4].repeat(ws_tex.shape[0], 1, 1, 1))  # (B, C, H, W)
#         img_edited = eval_get3d_angles(g_train, z_geo=ws_geo, z_tex=ws_tex, cameras=[generate_rotate_camera_list()[4]], intermediate_space=True)
#         w_loss = clip_loss.global_loss(img_edited).mean()

#         w_loss.backward()
#         w_optim.step()
#         w_optim.zero_grad()

#     tex_distance = (ws_tex - ws_tex_original).abs().mean(dim=-1).mean(dim=0)
#     geo_distance = (ws_geo - ws_geo_original).abs().mean(dim=-1).mean(dim=0)

#     cutoff = len(tex_distance)

#     chosen_layers_idx = torch.topk(torch.cat([tex_distance, geo_distance], dim=0), k)[1].tolist()

#     chosen_layer_idx_tex = []
#     chosen_layer_idx_geo = []
#     for idx in chosen_layers_idx:
#         if idx >= cutoff:
#             chosen_layer_idx_geo.append(idx - cutoff)
#         else:
#             chosen_layer_idx_tex.append(idx)

#     del ws_geo_original
#     del ws_geo
#     del ws_tex_original
#     del ws_tex
#     torch.cuda.empty_cache()

#     return chosen_layer_idx_tex, chosen_layer_idx_geo