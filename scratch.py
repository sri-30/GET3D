from clip_utils.clip_loss import CLIPLoss
import pickle
import torch



with open('latent_transform_adam_results/output_img_2_0.001_0.1_Sports Car_500.pickle', 'rb') as f:
    obj = pickle.load(f)[0]

clip_loss = CLIPLoss(text_prompt='Sports Car', target_type='PAE', clip_pae_args={'original_image': obj['Original'].to('cuda')})

x = clip_loss(obj['Original'][0].to('cuda'))
y = clip_loss(obj['Edited'][0].to('cuda'))

print(f'Cosine Similarity Loss: Original: {x} Edited: {y}')