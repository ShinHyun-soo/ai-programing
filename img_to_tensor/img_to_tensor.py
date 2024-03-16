import cv2
import torch

img1 = cv2.imread('imgs/1.jpg', cv2.COLOR_BGR2RGB)
img2 = cv2.imread('imgs/2.jpg', cv2.COLOR_BGR2RGB)
img3 = cv2.imread('imgs/3.jpg', cv2.COLOR_BGR2RGB)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img1_tensor = torch.FloatTensor(img1).to(device)
img2_tensor = torch.FloatTensor(img2).to(device)
img3_tensor = torch.FloatTensor(img3).to(device)

img1_tensor = img1_tensor.permute(2, 0, 1)
img2_tensor = img2_tensor.permute(2, 0, 1)
img3_tensor = img3_tensor.permute(2, 0, 1)

img_stack = torch.stack([img1_tensor, img2_tensor, img3_tensor], dim=0)

img_reshaped = img_stack.reshape(3, -1)

print(img_reshaped.mean(dim=1))