import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import clip
from submodules.CLIP_PAE.get3d_utils import get_pae
from torch.nn.functional import normalize

class CLIPLoss(torch.nn.Module):

    def __init__(self, text_prompt: str, target_type='text', clip_pae_args=None):
        super(CLIPLoss, self).__init__()
        self.model = clip.load("ViT-B/32", device="cpu")[0].to('cuda')
        self.target_type = target_type
        self.text_prompt = text_prompt
        self.text_tokenized = clip.tokenize(text_prompt).to('cuda')
        self.text_encoded = self.model.encode_text(torch.cat([clip.tokenize(text_prompt)]).cuda())
        self.cos_criterion = torch.nn.CosineSimilarity()
        if self.target_type == 'text':
            self.target = self.text_tokenized
        else:
            # Projected Embedding
            text_features = normalize(self.text_encoded)
            self.image_transformed = self.transform(clip_pae_args['original_image'])
            image_features = self.model.encode_image(self.image_transformed)
            self.target, self.basis = get_pae(self.model, n_components=10, image_features=image_features, text_features=text_features)
    
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
        else:
            image_processed = self.transform(image).unsqueeze(0)
            image_features = self.model.encode_image(image_processed)

            c_loss = (-1 * self.cos_criterion(image_features @ self.basis, self.target)).mean()
            print(c_loss)
            return c_loss