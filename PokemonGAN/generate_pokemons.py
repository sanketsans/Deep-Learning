from models import Discriminator, Generator
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def view_samples(sample):

    img = np.squeeze(sample, axis=0)
    img = img.detach().cpu().numpy()
    print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
    img = Image.fromarray(img, 'RGB')
    # print(img)
    img.show()

if __name__=="__main__":

    z_size = 100
    samples = []
    sample_size = 1

    D = Discriminator()
    G = Generator(z_size)

    dir = str(pathlib.Path().absolute()) + '/'

    G.load_state_dict(torch.load(dir + 'checkpoint_G.pth', map_location='cpu'))
    D.load_state_dict(torch.load(dir + 'checkpoint_D.pth', map_location='cpu'))

    G.eval()

    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    print(fixed_z.shape)
    sample = G(fixed_z)
    _ = view_samples(sample)
