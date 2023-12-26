import torch
import dnnlib
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import copy

import numpy as np
import pickle

from training.inference_utils import save_image_grid

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    G_ema.update_w_avg()
    camera_list = G_ema.synthesis.generate_rotate_camera_list()
    camera = camera_list[4]
    img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
        z=tex_z, geo_z=geo_z, c=grid_c, noise_mode='const',
        generate_no_light=True, truncation_psi=0.7, camera=camera)
    rgb_img = img[:, :3]
    return rgb_img

def preprocess_rgb(array):
    lo, hi = -1, 1
    img = array
    img = img.transpose(1, 3)
    img = img.transpose(1, 2)
    img = (img - lo) * (255 / (hi - lo))
    img.clip(0, 255)
    return img

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

def preprocess_clip(array, size):
    lo, hi = -1, 1
    img = array
    img = (img - lo) * (255 / (hi - lo))
    img.clip(0, 255)
    transform_clip = Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform_clip(img)

from clip_utils.clip_loss import CLIPLoss
from PIL import Image

def train(G, model, data_geo_z, data_tex_z, optimizer):
    model.train()
    loss_fn = CLIPLoss('SUV')

    cur_output = None

    latent_codes = []

    n_epochs = 15

    batch_size = 5

    lmbda = 0.0015

    for i in range(n_epochs):
        for j, g_z in enumerate(data_geo_z):
            grid_z = g_z.detach()
            grid_tex_z = data_tex_z[j].detach()
            g_ema = copy.deepcopy(G).eval()

            # Transform latents with model
            grid_z.requires_grad = True
            latents_edited = model(grid_z).reshape(1, 512)

            # Get output of GET3D on latents
            grid_c = torch.ones(1, device='cuda')
            output = eval_get3d_single(g_ema, latents_edited, grid_tex_z, grid_c)

            cur_output = output.detach()

            # Get CLIP Loss
            loss = loss_fn(output[0]) + lmbda * ((latents_edited - grid_z) ** 2).sum()

            print(loss)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i == n_epochs - 1:
                latent_codes.append((grid_z.detach(), latents_edited.detach()))
    
    latent_codes_output = []
    g_ema = copy.deepcopy(G).eval()
    for i, (original, edited) in enumerate(latent_codes):
        with torch.no_grad():
            img_original = eval_get3d_single(g_ema, original, data_tex_z[i], grid_c)
            img_edited = eval_get3d_single(g_ema, edited, data_tex_z[i], grid_c)
            latent_codes_output.append((img_original.cpu(), img_edited.cpu()))
    
    with open('output_codes.pickle', 'wb') as f:
        pickle.dump([(x.cpu(), y.cpu()) for x, y in latent_codes], f)
    
    with open('output_img.pickle', 'wb') as f:
        pickle.dump(latent_codes_output, f)
    


c = None
with open('test.pickle', 'rb') as f:
    c = pickle.load(f)

G_ema = constructGenerator(**c)

n_shape = 5

grid_c = torch.ones(n_shape, device=device)
grid_z = torch.randn([n_shape, 1, 512], device=device)  # random code for geometry
grid_tex_z = torch.randn([n_shape, 1, 512], device=device)  # random code for texture

# x = eval_get3d_tensor(G_ema, grid_z, grid_tex_z, grid_c)

model = TransformLatent().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 5

print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train(G_ema, model, grid_z, grid_tex_z, optimizer)




# c = None
# with open('test.pickle', 'rb') as f:
#     c = pickle.load(f)

# G_ema = constructGenerator(**c)

# n_shape = 5

# grid_c = torch.ones(n_shape, device=device)
# grid_z = torch.zeros([n_shape, 1, 512], device=device)  # random code for geometry
# grid_tex_z = torch.randn([n_shape, 1, 512], device=device)  # random code for texture

# x = eval_get3d_tensor(G_ema, grid_z, grid_tex_z, grid_c)

# with open('output6.pickle', 'wb') as f:
#     pickle.dump(x.cpu(), f)

# model = TransformLatent().to(device)

# learning_rate = 1e-3
# batch_size = 64
# epochs = 5

# print(model)

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train(G_ema, model, torch.zeros([n_shape, 512], device=device).split(1), optimizer)

# Get output of generator
# grid_size = (5, 5)
# n_shape = grid_size[0] * grid_size[1]

# grid_c = torch.ones(n_shape, device=device).split(1)
# grid_z = torch.zeros([n_shape, 512], device=device).split(1)  # random code for geometry
# grid_tex_z = torch.randn([n_shape, 512], device=device).split(1)  # random code for texture

# m = eval_get3d(G_ema, grid_z, grid_tex_z, grid_c)

# img = save_image_grid(m, None, [-1, 1], grid_size)

# with open('output4.pickle', 'wb') as f:
#     pickle.dump(img, f)