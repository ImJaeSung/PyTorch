# image : torchvision, imgaug library
import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms

transform = transforms.Compose( # Compose 클래스로 통합
    [
        transforms.Resize(size = (512, 512)), # 512x512 크기로 변환
        transforms.ToTensor() # PIL.Image형식을 Tensor형식으로 변환
    ]
) 

"""텐서화 클래스"""
# 0~255 범위의 픽셀값을 0~1사이의 값으로 min-max normalization
# H, W, C -> C, H, W

image = Image.open('../archive/datasets/images/cat.jpg')
transformed_image = transform(image)
print(transformed_image.shape) # C, H, W

"""Rotation and Flip"""
# degree : sequence 형태로 범위를 전달하여 random 회전도 가능
# expand = True : 여백이 없어짐
# center : default 왼쪽상단을 기준으로 회전
transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees = 30, expand = False, center = None),
        transforms.RandomHorizontalFlip(p = 0.5), # 0.5의 확률로 대칭 수행
        transforms.RandomVerticalFlip(p = 0.5)
    ]
)

transform(image)

"""Cutting and Padding"""
transform = transforms.Compose(
    [
        transforms.RandomCrop(size = (512, 512)), # HxW
        transforms.Pad(padding = 50, fill = (127, 127, 255), padding_mode = "constant") # symmeteric, reflect
        # 512x512 -> 612x612
        # 테두리 color : RGB(127, 127, 255)
    ]
)

transform(image)

"""Resize"""
transform = transforms.Compose(
    [
        transforms.Resize(size = (512, 512))
    ]
)
transform(image)

transform = transforms.Compose(
    [
        transforms.Resize(size = (500)) # H,W 중 더 작은 값에 비율을 맞춰 크기 수정
    ]
)

transform(image)

"""geometric tranform"""
transform = transforms.Compose(
    [
        transforms.RandomAffine(
            degrees = 15, translate = (0.2, 0.2),
            scale = (0.8, 1.2), shear = 15
        )
    ]
)

transform(image)

"""color"""
transform = transforms.Compose(
    [
        transforms.ColorJitter(
            brightness= 0.3, contrast = 0.3,
            saturation = 0.3, hue = 0.3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        transforms.ToPILImage()
    ]
)
transform(image)

transform = transforms.Compose(
    [
        transforms.ColorJitter(
            brightness= 0.3, contrast = 0.3,
            saturation = 0.3, hue = 0.3
        ),
    ]
)

transform(image)

"""Noise"""
class IaaTransforms:
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.SaltAndPepper(p = (0.03, 0.07)),
            iaa.Rain(speed = (0.3, 0.7))
        ])

    def __call__(self, images):
        images = np.array(images)
        augmented = self.seq.augment_image(images)
        return Image.fromarray(augmented)

transform = transforms.Compose([    
    IaaTransforms()
])

transform(image)