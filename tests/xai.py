import torch
import models
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Saliency, GuidedGradCam, DeepLift

device = torch.device('cpu')

# Create models
generator = models.GeneratorMNIST().to(device)
discriminator = models.DiscriminatorMNIST().to(device)

# Load weights (model.load_state_dict(torch.load(PATH)))
generator.load_state_dict(torch.load('weights/gen_epoch_7.pth', map_location=torch.device(device)))
discriminator.load_state_dict(torch.load('weights/disc_epoch_7.pth', map_location=torch.device(device)))

# Generate a fake sample
noise = torch.randn(1, 100, 1, 1, device=device)

fake = generator(noise)

img = fake.detach().numpy()
img = np.squeeze(img)

plt.imsave('xai-output/fake.png', img, cmap='gray')

# Saliency
saliency = Saliency(discriminator)
grads = saliency.attribute(fake)
grads = grads.squeeze().cpu().detach().numpy()

plt.imsave('xai-output/saliency.png', grads)

# Guided Grad-CAM
ggc = GuidedGradCam(discriminator, discriminator.network[9])
grads = ggc.attribute(fake)
grads = grads.squeeze().cpu().detach().numpy()
print(np.shape(grads))

plt.imsave('xai-output/gradcam.png', grads)

# DeepLift
dl = DeepLift(discriminator)
att = dl.attribute(fake)
att = att.squeeze().cpu().detach().numpy()

plt.imsave('xai-output/deeplift.png', att)
