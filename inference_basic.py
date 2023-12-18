# Generate Seed
# Generate Latent Codes
# Generate Model

from training import inference_3d
import pickle

c = None
with open('test.pickle', 'rb') as f:
    c = pickle.load(f)
print(c)
inference_3d.inference(rank=0, **c)