import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import clip

class CLIPLoss(torch.nn.Module):

    def __init__(self, text_prompt: str, target_type='text', clip_pae_args=None):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.target_type = target_type
        self.text_prompt = text_prompt
        self.text_encoded = clip.tokenize(text_prompt).to('cuda')
        self.text_encoded.requires_grad = False
        if self.target_type == target_type:
            self.target = self.text_encoded
        else:
            self.target = None
        
    
    def transform(self, array):
        lo, hi = -1, 1
        img = array
        img = (img - lo) * (1 / (hi - lo))
        img.clip(0, 1)
        transform_clip = Compose([
            Resize(self.model.visual.input_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.model.visual.input_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return transform_clip(img)

    def forward(self, image):
        if self.target_type == 'text':
            image_processed = self.transform(image).unsqueeze(0)
            similarity = 1 - self.model(image_processed, self.target)[0] / 100
            return similarity