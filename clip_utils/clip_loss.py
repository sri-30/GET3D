import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import clip

class CLIPLoss(torch.nn.Module):

    def __init__(self, text_prompt: str):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.text_prompt = text_prompt
        self.text_encoded = clip.tokenize(text_prompt).to('cuda')
        self.text_encoded.requires_grad = False
        
    
    def transform(self, array):
        lo, hi = -1, 1
        img = array
        img = (img - lo) * (255 / (hi - lo))
        img.clip(0, 255)
        transform_clip = Compose([
            Resize(self.model.visual.input_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.model.visual.input_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return transform_clip(img)

    def forward(self, image):
        image_processed = self.transform(image).unsqueeze(0)
        similarity = 1 - self.model(image_processed, self.text_encoded)[0] / 100
        return similarity