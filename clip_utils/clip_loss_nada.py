import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import clip
#from submodules.CLIP_PAE.get3d_utils import get_pae
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DEVICE='cuda'

# project v onto the direction of u
def project_to_vector(v, u):
    return (u.dot(v) / u.dot(u)) * u.clone()


# perform Gram-Schmidt process to make a set of vectors orthonormal
# each row of vv is a given vector
@torch.no_grad()
def gram_schmidt(vv):
    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device, dtype=vv.dtype)
    uu[0] += vv[0]
    for k in range(1, nk):
        uu[k] += vv[k]
        for j in range(0, k):
            uu[k] -= project_to_vector(vv[k], uu[j])
        uu[k] /= uu[k].norm()
    return uu

def get_pae_PCA_basis(n_components=10, embeddings=[]):
    # No precomputed basis. Compute now
    all_embeddings = embeddings
    type_before = all_embeddings.dtype
    all_embeddings = StandardScaler().fit_transform(all_embeddings.cpu().numpy())
    pca = PCA(n_components=n_components)
    pca.fit(all_embeddings)
    basis = torch.from_numpy(pca.components_).to(DEVICE).to(type_before)
    return basis


@torch.no_grad()
def get_pae(model, args, image_features, text_features, all_texts):
    semantic_basis_text = all_texts
    subspace_basis = clip.tokenize(semantic_basis_text).to(DEVICE)
    subspace_basis = model.encode_text(subspace_basis)

    if "GS" in args['target']:
        subspace_basis = F.normalize(subspace_basis)
        subspace_basis = gram_schmidt(subspace_basis)
        text_coeff_sum = (text_features @ subspace_basis.T).sum(dim=-1)
    elif "PCA" in args['target']:
        subspace_basis = get_pae_PCA_basis(n_components=args['components'], embeddings=subspace_basis)
        text_coeff_sum = (text_features @ subspace_basis.T).sum(dim=-1)
    elif "None" in args['target']:
        subspace_basis = F.normalize(subspace_basis)
        text_coeff_sum = torch.ones(text_features.shape[0], device=DEVICE, dtype=text_features.dtype)
    else:
        subspace_basis = F.normalize(subspace_basis)
        text_coeff_sum = (text_features @ subspace_basis.T).sum(dim=-1)
        print("yes")
    image_coeff = image_features @ subspace_basis.T
    image_coeff_abs = args['power'] * abs(image_coeff)

    # augmentation
    targets = []
    for i, text_feature in enumerate(text_features):
        for j, image_feature in enumerate(image_features):
            target = image_feature.clone()
            if "Ex" in args['target']:
                target = image_feature + args['power'] * text_feature - to_deduct_per_img[j]
            elif "+" in args['target']:
                shift = 0
                for k, basis_vector in enumerate(subspace_basis):
                    coeff = image_coeff_abs[j][k]
                    shift += coeff
                    target -= image_coeff_abs[j][k] * basis_vector
                target += (shift / text_coeff_sum[i]) * text_feature
            else: # pae
                target += args['power'] * text_feature
            targets.append(target)
    targets = torch.stack(targets)
    return targets

class CLIPLoss(torch.nn.Module):

    def __init__(self, source_text, target_text, aux_string='', corpus=[], method='All', original_images=None):
        super(CLIPLoss, self).__init__()
        self.model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self.preprocess = Compose([Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +
                                              clip_preprocess.transforms[:2] +
                                              clip_preprocess.transforms[4:])

        # self.model_, clip_preprocess_ = clip.load("ViT-B/16", device="cuda")
        # self.preprocess_ = Compose([Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +
        #                                       clip_preprocess_.transforms[:2] +
        #                                       clip_preprocess_.transforms[4:])

        self.source_text = aux_string + source_text.lower()
        self.target_text = aux_string + target_text.lower()
        with torch.no_grad():
            self.source_tokenized = clip.tokenize(source_text).to('cuda')
            self.source_text_encoded = self.model.encode_text(clip.tokenize(source_text).cuda())
            self.target_tokenized = clip.tokenize(target_text).to('cuda')
            self.target_text_encoded = self.model.encode_text(clip.tokenize(target_text).cuda())
            self.direction = (self.target_text_encoded - self.source_text_encoded)
            self.direction  = self.direction/self.direction.clone().norm(dim=-1, keepdim=True)

            # source_text_encoded_ = self.model_.encode_text(clip.tokenize(source_text).cuda())
            # target_text_encoded_ = self.model_.encode_text(clip.tokenize(target_text).cuda())
            # direction_ = (target_text_encoded_ - source_text_encoded_)
            # self.direction_  = direction_/direction_.clone().norm(dim=-1, keepdim=True)

            if corpus:
                # Convert corpus to CLIP text embeddings
                self.corpus_text = [f'{aux_string}{word.lower()}' for word in corpus]
                self.corpus_tokenized = clip.tokenize(self.corpus_text).to('cuda')
                self.corpus_embeddings = self.model.encode_text(self.corpus_tokenized)
                # Convert corpus to basis using PCA or GS
                # from sklearn.decomposition import PCA
                # pca = PCA(n_components=4)
                # pca.fit(self.corpus_embeddings.cpu().numpy())
                # self.components = torch.from_numpy(pca.components_).to('cuda').type(torch.float16)
                self.components = self.corpus_embeddings
                self.components = (self.components)

                source_text_projected = (self.source_text_encoded @ self.components.T)
                target_text_projected = (self.target_text_encoded @ self.components.T)

                self.direction_projected = F.normalize(target_text_projected - source_text_projected)

                # Project target text embedding onto basis
                self.projection_target = (F.normalize(self.target_text_encoded) @ self.components.T).softmax(dim=-1)
                if not original_images is None:
                    self.original_images_embeddings = self.model.encode_image(self.preprocess(original_images))
                    self.pae_targets = get_pae(self.model, {'power': 6, 'target': 'None+', 'components': len(corpus)}, self.original_images_embeddings, F.normalize(self.target_text_encoded), corpus)

    def projection_embedding_loss(self, image):
        # Get image embedding
        image_processed = self.preprocess(image)
        image_embeddings = F.normalize(self.model.encode_image(image_processed))

        # Project image embeddings onto basis
        similarity_scores = (image_embeddings @ self.components.T).softmax(dim=-1)
        return 1. - F.cosine_similarity(similarity_scores, self.projection_target).mean()

    def directional_pae_loss(self, source_image, target_image):
        pass

    def directional_projection_loss(self, source_image, target_image):
        source_preprocessed = self.preprocess(source_image)
        target_preprocessed = self.preprocess(target_image)
        
        source_encoded = self.model.encode_image(source_preprocessed)
        target_encoded = self.model.encode_image(target_preprocessed)

        source_projected = (source_encoded @ self.components.T)
        target_projected = (target_encoded @ self.components.T)

        img_direction = F.normalize(target_projected - source_projected)
        return 1. - F.cosine_similarity(img_direction, self.direction_projected)

    
    def projection_augmentation_loss(self, image):
        image_processed = self.preprocess(image)
        image_encoded = self.model.encode_image(image_processed)
        sim_scores = torch.zeros((image_encoded.shape[0]))
        for i, _ in enumerate(image_encoded):
            sim_scores[i] = torch.dot(image_encoded[i], self.pae_targets[i])/(image_encoded[i].norm() * self.pae_targets[i].norm())
        return 1 - sim_scores.mean()

    def projection_augmentation_loss_nada(self, image_original, image_target):
        image_original_processed = self.preprocess(image_original)
        image_original_encoded = self.model.encode_image(image_original_processed)

        image_target_processed = self.preprocess(image_target)
        image_target_encoded = self.model.encode_image(image_target_processed)

        pae = get_pae(self.model, {'power': 8, 'target': 'None+', 'components': len(self.corpus_text)}, image_original_encoded, F.normalize(self.target_text_encoded), self.corpus_text)
        sim_scores = torch.zeros((image_target_encoded.shape[0]))
        for i, _ in enumerate(image_target_encoded):
            sim_scores[i] = torch.dot(image_target_encoded[i], pae[i])/(image_target_encoded[i].norm() * pae[i].norm())
        return 1 - sim_scores.mean()

    def get_embedding_image(self, image):
        image_processed = self.preprocess(image)
        return self.model.encode_image(image_processed)
    
    def global_loss(self, image):
        image_processed = self.preprocess(image)
        logits_per_image, _ = self.model(image_processed, self.target_tokenized)
        return (1. - logits_per_image / 100).mean()
    
    def directional_loss(self, source_image, target_image):
        source_preprocessed = self.preprocess(source_image)
        target_preprocessed = self.preprocess(target_image)
        source_encoded = self.model.encode_image(source_preprocessed)
        target_encoded = self.model.encode_image(target_preprocessed)
        img_direction = target_encoded - source_encoded
        img_direction = img_direction/img_direction.clone().norm(dim=-1, keepdim=True)
        return 1. - F.cosine_similarity(img_direction, self.direction)

    def directional_loss_(self, source_image, target_image):
        source_preprocessed = self.preprocess_(source_image)
        target_preprocessed = self.preprocess_(target_image)
        source_encoded = self.model_.encode_image(source_preprocessed)
        target_encoded = self.model_.encode_image(target_preprocessed)
        img_direction = target_encoded - source_encoded
        img_direction = img_direction/img_direction.clone().norm(dim=-1, keepdim=True)
        return 1. - F.cosine_similarity(img_direction, self.direction)

    def forward(self, source_image, target_image=None):
        # return self.directional_loss(source_image, target_image) + self.directional_loss_(source_image, target_image)
        source_preprocessed = self.preprocess(source_image)
        target_preprocessed = self.preprocess(target_image)
        source_encoded = self.model.encode_image(source_preprocessed)
        target_encoded = self.model.encode_image(target_preprocessed)
        img_direction = target_encoded - source_encoded
        img_direction = img_direction/img_direction.clone().norm(dim=-1, keepdim=True)
        return 1. - F.cosine_similarity(img_direction, self.direction)
            #return ((1 - torch.nn.MSELoss()(img_direction, self.direction)) + (1 - self.cos_sim(img_direction, self.direction)))/2