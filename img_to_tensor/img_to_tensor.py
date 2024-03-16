import cv2
import torch

# opencv 로 사진 읽어 오기
img1 = cv2.imread('imgs/1.jpg', cv2.COLOR_BGR2RGB)
img2 = cv2.imread('imgs/2.jpg', cv2.COLOR_BGR2RGB)
img3 = cv2.imread('imgs/3.jpg', cv2.COLOR_BGR2RGB)

# cuda를 사용할 수 있으면 device 를 cuda, 못하면 cpu 로 설정.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# opencv 로 읽어온 행렬들을 FloatTensor로 GPU에 저장.
img1_tensor = torch.FloatTensor(img1).to(device)
img2_tensor = torch.FloatTensor(img2).to(device)
img3_tensor = torch.FloatTensor(img3).to(device)

# [H, W, C] to [C, H, W]
img1_tensor = img1_tensor.permute(2, 0, 1)
img2_tensor = img2_tensor.permute(2, 0, 1)
img3_tensor = img3_tensor.permute(2, 0, 1)

# [C, H, W], [C, H, W], [C, H, W] => [3, C, H, W]
img_stack = torch.stack([img1_tensor, img2_tensor, img3_tensor], dim=0)

# [3, C, H, W] => [3, C*H*W]
img_reshaped = img_stack.reshape(3, -1)

# 각 차원들의 평균 계산
print(img_reshaped.mean(dim=1))