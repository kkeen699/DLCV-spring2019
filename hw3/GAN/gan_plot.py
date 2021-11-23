import torch
import torchvision
import os
import sys
from gan_model import Generator

file_root = sys.argv[1]
file_root = os.path.join(file_root, 'fig1_2.jpg')

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

generator = Generator().to(device)
generator.load_state_dict(torch.load('./models/gan.pth', map_location=device))

torch.manual_seed(6)
fixed_noise = torch.randn(32, 100, 1, 1).to(device)

generator.eval()
img = generator(fixed_noise)
torchvision.utils.save_image(img.cpu().data, file_root, nrow=8)