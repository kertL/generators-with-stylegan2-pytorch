import PIL.Image
import torch
from torchvision.utils import save_image
import os
import dnnlib

import pretrained_networks


def text_save(file, data):  # save generate code, which can be modified to generate customized style
    for i in range(len(data[0])):
        s = str(float(data[0][i]))+'\n'
        file.write(s)

def generate_images(network_pkl, num, truncation_psi=0.5, device='cuda'):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl, device=device)
    
    for i in range(num):
        print('Generating image %d/%d ...' % (i+1, num))
        
        z = torch.randn(1, Gs.z_dim)
        
        txt_filename = 'results/generate_codes/' + str(i).zfill(4) + '.txt'
        with open(txt_filename, 'w') as f:
            text_save(f, z)
        z = z.to(device)

        # 生成图像
        with torch.no_grad():
            images = Gs(z, None, truncation_psi=truncation_psi, noise_mode='const')  # 注意调用方式和参数可能有所不同

        image_path = 'results/' + str(i) + '.png'
        save_image((images + 1) / 2, image_path) # 调整图像范围并保存


def main():
    device = 'cuda'
    os.makedirs('results/', exist_ok=True)
    os.makedirs('results/generate_codes/', exist_ok=True)

    network_pkl = 'networks/generator_star-stylegan2-config-f.pkl'  # 不用转换，原版模型
    generate_num = 20  # 生成数量

    generate_images(network_pkl, generate_num, device=device)

if __name__ == "__main__":
    main()