import torch
import PIL.Image
import numpy as np
from legacy import load_network_pkl  
import os

def read_feature(file_name, device='cuda'):
    # 加载特征
    with open(file_name, mode='r') as file:
        contents = file.readlines()
    code = [float(line.strip('\n')) for line in contents]
    return torch.tensor(code, dtype=torch.float32, device=device).unsqueeze(0)

def move_latent_and_save(latent_vector, direction_file, coeffs, G, device='cuda'):
    # 加载方向文件
    direction_np = np.load('latent_directions/' + direction_file)
    direction = torch.from_numpy(direction_np).float().to(device)
    os.makedirs('results/' + direction_file.split('.')[0], exist_ok=True)
    for i, coeff in enumerate(coeffs):
        print('Generating image %d/%d ...' % (i+1, len(coeffs)))
        new_latent_vector = latent_vector.clone()
        new_latent_vector[:, :8] += coeff * direction[:8]

        # 使用G生成图像
        with torch.no_grad():
            images = G.synthesis(new_latent_vector, noise_mode='const')
        # 转换图像格式并保存
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        result = PIL.Image.fromarray(images[0].cpu().numpy(), 'RGB')
        result.save(f'results/{direction_file.split(".")[0]}/{str(i).zfill(3)}.png')

def main():
    # 加载模型
    device = 'cuda'
    with open('networks/generator_star-stylegan2-config-f.pkl', "rb") as f:
        G = load_network_pkl(f)['G_ema'].to(device)

    # 选择潜在向量
    state_dict = G.state_dict()
    w_avg_key = [key for key in state_dict if 'w_avg' in key]
    if w_avg_key:
        w_avg = state_dict[w_avg_key[0]]
    else:
        raise AttributeError("模型中未找到w_avg变量")

    face_latent = read_feature('results/generate_codes/0015.txt', device)
    w = G.mapping(face_latent, None)
    truncation_psi = 0.5
    w = w_avg + (w - w_avg) * truncation_psi

    # 选择调整方向和大小
    direction_file = 'angle_horizontal.npy'
    coeffs = [-15., -12., -9., -6., -3., 0., 3., 6., 9., 12.]

    # 开始调整并保存图片
    move_latent_and_save(w, direction_file, coeffs, G, device)

if __name__ == "__main__":
    main()