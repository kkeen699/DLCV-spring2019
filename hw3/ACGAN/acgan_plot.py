import torch
import torchvision
import os
import sys
import numpy as np
from acgan_model import Generator

file_root = sys.argv[1]
file_root = os.path.join(file_root, 'fig2_2.jpg')

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

generator = Generator().to(device)
generator.load_state_dict(torch.load('./models/acgan.pth', map_location=device))

torch.manual_seed(6)
up = np.ones(10)
down = np.zeros(10)
fixed_class = np.hstack((up,down))
fixed_class = torch.from_numpy(fixed_class).view(20,1,1,1).type(torch.FloatTensor)
fixed_noise = torch.randn(10, 100, 1, 1)
fixed_noise = torch.cat((fixed_noise,fixed_noise))
fixed_noise = torch.cat((fixed_noise, fixed_class),1).to(device)

generator.eval()
img = generator(fixed_noise)
torchvision.utils.save_image(img.cpu().data, file_root, nrow=10)