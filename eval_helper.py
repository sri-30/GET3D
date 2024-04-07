import subprocess

random_seed_limit = 50

text_prompts = ['Sports Car', 'SUV', 'Hatchback', 'Sedan']

for random_seed in range(random_seed_limit):
    for text_prompt in text_prompts:
        subprocess.run(["python", "latent_optimization.py", f'{random_seed}', text_prompt, "global", f'{0.0001}', f'{0.01}'])
        subprocess.run(["python", "latent_optimization.py", f'{random_seed}', text_prompt, "pae", f'{0.0001}', f'{0.01}'])