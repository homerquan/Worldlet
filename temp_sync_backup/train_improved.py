import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

# Import components
from rl_doom import DoomEnv
from vq_vae import VQVAE
from transformer_world import TransformerWorldModel
from torch.amp import autocast, GradScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RES = 64
SCENARIOS = ['basic', 'deadly_corridor', 'defend_the_center', 'defend_the_line', 'health_gathering']

def collect_diverse_data(total_samples=50000):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    print(f'Collecting {total_samples} samples across {len(SCENARIOS)} scenarios sequentially...')
    
    samples_per_scenario = total_samples // len(SCENARIOS)
    
    frames = []
    actions = []
    next_frames = []
    
    samples_collected = 0
    start_time = time.time()
    
    for scenario in SCENARIOS:
        print(f'Starting scenario: {scenario}')
        env = DoomEnv(render=False, scenario=scenario)
        obs, _ = env.reset()
        
        # Momentum random walk to explore
        current_action = env.action_space.sample()
        action_duration = np.random.randint(1, 15)
        
        for _ in range(samples_per_scenario):
            action_duration -= 1
            if action_duration <= 0:
                current_action = env.action_space.sample()
                action_duration = np.random.randint(1, 15)
                
            next_obs, reward, done, trunc, _ = env.step(current_action)
            
            f1 = cv2.resize(obs, (RES, RES))
            f2 = cv2.resize(next_obs, (RES, RES))
            
            frames.append(f1)
            actions.append(current_action)
            next_frames.append(f2)
            
            samples_collected += 1
            if samples_collected % 5000 == 0:
                fps = samples_collected / (time.time() - start_time)
                print(f'Collected {samples_collected}/{total_samples} samples (FPS: {fps:.1f})')
                
            obs = next_obs
            if done or trunc:
                obs, _ = env.reset()
                
        env.close()
    
    os.makedirs('data', exist_ok=True)
    filename = 'data/transitions_doom_diverse.npz'
    np.savez(filename, frames=np.array(frames), actions=np.array(actions), next_frames=np.array(next_frames))
    print(f'Diverse dataset saved to {filename}')

def train_vqvae(epochs=10, batch_size=256):
    print(f'Training VQ-VAE on diverse dataset...')
    dataset = np.load('data/transitions_doom_diverse.npz')
    # Use both current and next frames to get more data
    all_frames = np.concatenate([dataset['frames'], dataset['next_frames']], axis=0)
    all_frames = torch.tensor(all_frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    
    model = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    
    dataset_size = len(all_frames)
    
    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        epoch_loss = 0
        for i in range(0, dataset_size, batch_size):
            idx = perm[i:i+batch_size]
            b_frames = all_frames[idx].to(DEVICE)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                recon, vq_loss, _ = model(b_frames)
                recon_loss = F.mse_loss(recon, b_frames)
                loss = recon_loss + vq_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * len(idx)
            
        print(f'VQ-VAE Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/dataset_size:.5f}')
        
    save_file(model.state_dict(), 'models/vq_vae_doom.safetensors')
    print('VQ-VAE saved.')
    return model

def encode_data(vqvae, batch_size=512):
    print('Encoding dataset with trained VQ-VAE...')
    dataset = np.load('data/transitions_doom_diverse.npz')
    frames = torch.tensor(dataset['frames'], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    next_frames = torch.tensor(dataset['next_frames'], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    
    encoded_f = []
    encoded_n = []
    
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            f_b = frames[i:i+batch_size].to(DEVICE)
            n_b = next_frames[i:i+batch_size].to(DEVICE)
            
            with autocast('cuda'):
                _, _, _, enc_f = vqvae.vq(vqvae.encoder(f_b))
                _, _, _, enc_n = vqvae.vq(vqvae.encoder(n_b))
                
            encoded_f.append(enc_f.cpu())
            encoded_n.append(enc_n.cpu())
            
    encoded_f = torch.cat(encoded_f, dim=0).squeeze(-1).numpy()
    encoded_n = torch.cat(encoded_n, dim=0).squeeze(-1).numpy()
    actions = dataset['actions']
    
    np.savez('data/encoded_diverse.npz', frames=encoded_f, actions=actions, next_frames=encoded_n)
    print('Encoded dataset saved.')

def train_transformer(epochs=20, batch_size=256):
    print('Training Deeper Transformer World Model...')
    dataset = np.load('data/encoded_diverse.npz')
    f_tokens = torch.tensor(dataset['frames'], dtype=torch.long).view(-1, 64)
    a_tokens = torch.tensor(dataset['actions'], dtype=torch.long).unsqueeze(-1) + 512
    n_tokens = torch.tensor(dataset['next_frames'], dtype=torch.long).view(-1, 64)
    
    inputs = torch.cat([f_tokens, a_tokens, n_tokens], dim=1)[:, :-1]
    targets = torch.cat([f_tokens, a_tokens, n_tokens], dim=1)[:, 1:]
    
    # INCREASED CAPACITY
    model = TransformerWorldModel(vocab_size=512+5, d_model=512, nhead=8, num_layers=8, max_seq_len=129).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset_size = len(inputs)
    
    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        epoch_loss = 0
        for i in range(0, dataset_size, batch_size):
            idx = perm[i:i+batch_size]
            b_in = inputs[idx].to(DEVICE)
            b_tgt = targets[idx].to(DEVICE)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(b_in)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), b_tgt.reshape(-1))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * len(idx)
            
        scheduler.step()
        print(f'Transformer Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/dataset_size:.5f} - LR: {scheduler.get_last_lr()[0]:.6f}')
        
    save_file(model.state_dict(), 'models/transformer_world_doom.safetensors')
    print('Transformer saved.')

if __name__ == '__main__':
    collect_diverse_data(total_samples=50000)
    vqvae = train_vqvae(epochs=10)
    encode_data(vqvae)
    train_transformer(epochs=20)
