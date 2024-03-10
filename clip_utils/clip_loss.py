import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import clip
from torch.nn.functional import normalize
from submodules.CLIP_PAE.get3d_utils import get_pae

class CLIPLoss(torch.nn.Module):

    model_global = None

    def __init__(self, text_prompt: str, target_type='text', clip_pae_args=None):
        super(CLIPLoss, self).__init__()
        if CLIPLoss.model_global is None:
            CLIPLoss.model_global = clip.load("ViT-B/32", device="cpu")[0].to('cuda')
        self.model = CLIPLoss.model_global
        self.target_type = target_type
        self.text_prompt = text_prompt
        self.text_tokenized = clip.tokenize(text_prompt).to('cuda')
        self.text_encoded = self.model.encode_text(torch.cat([clip.tokenize(text_prompt)]).cuda())
        self.cos_sim = torch.nn.CosineSimilarity()
        if self.target_type == 'text':
            self.target = self.text_tokenized
        elif 'pae' in self.target_type:
            # Projected Embedding
            text_features = normalize(self.text_encoded)
            self.image_transformed = self.transform(clip_pae_args['original_image'])
            image_features = self.model.encode_image(self.image_transformed)
            self.target, self.basis = get_pae(image_features=image_features, text_features=text_features, power=clip_pae_args['power'], clip_target=clip_pae_args['clip_target'])
        elif 'directional' in self.target_type:
            with torch.no_grad():
                text_original = clip.tokenize(clip_pae_args['source_text'])
                text_original_encoded = self.model.encode_text(text_original.cuda())
                self.direction = (self.text_encoded - text_original_encoded)
                self.direction  = self.direction/self.direction.clone().norm(dim=-1, keepdim=True)
                self.image_original = self.transform(clip_pae_args['original_image'])
                self.image_original_encoded = self.model.encode_image(self.image_original)
    
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

    def get_embedding_image(self, image):
        if self.target_type == 'text':
            image_processed = self.transform(image).unsqueeze(0)
            return self.model.encode_image(image_processed)
        else:
            image_processed = self.transform(image).unsqueeze(0)
            image_features = self.model.encode_image(image_processed)
            return image_features @ self.basis
    
    def get_embedding_target(self, image):
        if self.target_type == 'text':
            return self.text_encoded
        return self.target
            

    def forward(self, image):
        if self.target_type == 'text':
            image_processed = self.transform(image).unsqueeze(0)
            similarity = 1 - self.model(image_processed, self.text_tokenized)[0] / 100
            return similarity
        elif 'pae' in self.target_type:
            image_processed = self.transform(image).unsqueeze(0)
            image_features = self.model.encode_image(image_processed)

            # c_loss = (-1 * self.cos_criterion(image_features @ self.basis, self.target)).mean()

            c_loss = -1 * (self.cos_sim(image_features @ self.basis, self.target)).mean()
            return c_loss
        elif 'directional' in self.target_type:
            image_processed = self.transform(image).unsqueeze(0)
            image_features = self.model.encode_image(image_processed)
            img_direction = image_features - self.image_original_encoded
            img_direction = img_direction/img_direction.clone().norm(dim=-1, keepdim=True)
            
            #return 1 - self.cos_sim((image_features - self.image_original_encoded), self.direction)
            return ((1 - torch.nn.MSELoss()(img_direction, self.direction)) + (1 - self.cos_sim(img_direction, self.direction)))/2