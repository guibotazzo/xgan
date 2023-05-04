# https://discuss.pytorch.org/t/how-to-calculate-the-jacobian-matrix-of-a-neural-network/153376/4
import torch
from torch.autograd.functional import jacobian
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open('fish.jpg')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_tensor = transform(image)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.to(torch.device("mps"))

model = resnet50(weights=ResNet50_Weights.DEFAULT).to(torch.device("mps"))

# 1) TOP-k SELECTION

# 2) GRADIENTS CALCULATION
J = jacobian(model, img_tensor)
A = img_tensor * J

A = A.squeeze()
print(A.size())
A = torch.mean(A, dim=0)
A = torch.reshape(A, (287, 426, 3))
A = A.cpu().numpy()

plt.imshow(A)
plt.show()

# 3) CO-VARIATION CALCULATION
