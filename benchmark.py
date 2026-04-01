import os
import torch
import numpy as np
import time
from rl_doom import DoomEnv
from video_world_model import VideoWorldModel
from transformer_world import TransformerWorldModel
from vq_vae import VQVAE
from safetensors.torch import load_file

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RES = 64

def evaluate_model(model_type, num_steps=1000, scenario='basic'):
    print(f'\n--- Benchmarking {model_type} on {scenario} for {num_steps} steps ---')
    env = DoomEnv(render=False, scenario=scenario)
    obs, _ = env.reset()
    
    if model_type == 'continuous':
        model = VideoWorldModel(num_actions=5).to(DEVICE)
        model_path = 'models/video_predict_model_doom.safetensors'
        if os.path.exists(model_path):
            model.load_state_dict(load_file(model_path, device=str(DEVICE)))
        model.eval()
    elif model_type == 'transformer':
        vqvae = VQVAE().to(DEVICE)
        vq_path = 'models/vq_vae_doom.safetensors'
        if os.path.exists(vq_path):
            vqvae.load_state_dict(load_file(vq_path, device=str(DEVICE)))
        vqvae.eval()
        
        # Make sure architecture matches our improved version
        model = TransformerWorldModel(vocab_size=512 + 5, d_model=512, nhead=8, num_layers=8, max_seq_len=129).to(DEVICE)
        tf_path = 'models/transformer_world_doom.safetensors'
        if os.path.exists(tf_path):
            model.load_state_dict(load_file(tf_path, device=str(DEVICE)))
        model.eval()
        
    mse_errors = []
    
    start_time = time.time()
    
    import cv2
    current_frame = cv2.resize(obs, (RES, RES))
    
    for step in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, done, trunc, _ = env.step(action)
        true_next_frame = cv2.resize(next_obs, (RES, RES))
        
        frame_t = (torch.tensor(current_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0).to(DEVICE)
        action_t = torch.tensor([action], dtype=torch.long, device=DEVICE)
        true_next_t = (torch.tensor(true_next_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0).to(DEVICE)
        
        with torch.no_grad():
            if model_type == 'continuous':
                pred_next_t = model(frame_t, action_t)
            elif model_type == 'transformer':
                z = vqvae.encoder(frame_t)
                _, _, _, enc_indices = vqvae.vq(z)
                current_tokens = enc_indices.view(1, 64)
                act_token = torch.tensor([[action + 512]], dtype=torch.long, device=DEVICE)
                context = torch.cat([current_tokens, act_token], dim=1)
                
                for _ in range(64):
                    logits = model(context)
                    next_token_logits = logits[:, -1, :512]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    context = torch.cat([context, next_token], dim=1)
                
                next_frame_tokens = context[:, 65:]
                flat_tokens = next_frame_tokens.view(-1, 1)
                encodings = torch.zeros(flat_tokens.shape[0], vqvae.vq._num_embeddings, device=DEVICE)
                encodings.scatter_(1, flat_tokens, 1)
                quantized = torch.matmul(encodings, vqvae.vq._embedding.weight)
                quantized = quantized.view(1, 8, 8, 64).permute(0, 3, 1, 2).contiguous()
                pred_next_t = vqvae.decoder(quantized)
                
        mse = torch.nn.functional.mse_loss(pred_next_t, true_next_t).item()
        mse_errors.append(mse)
        
        current_frame = true_next_frame
        
        if done or trunc:
            obs, _ = env.reset()
            current_frame = cv2.resize(obs, (RES, RES))
            
    end_time = time.time()
    
    avg_mse = np.mean(mse_errors)
    fps = num_steps / (end_time - start_time)
    
    print(f'Average MSE: {avg_mse:.5f}')
    print(f'Throughput: {fps:.2f} FPS')
    
    return avg_mse

if __name__ == '__main__':
    evaluate_model('transformer', 200, 'deadly_corridor')
    evaluate_model('continuous', 200, 'deadly_corridor')
