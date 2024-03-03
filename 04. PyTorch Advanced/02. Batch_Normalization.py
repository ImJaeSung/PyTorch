import torch
import torch.nn as nn

x = torch.FloatTensor(
    [
        [-0.6577, -0.5797, 0.6360],
        [0.7392, 0.2145, 1.523],
        [0.2432, 0.5662, 0.322]
    ]
)

# batch normalization
print(nn.BatchNorm1d(3)(x)) # 2D/3D 입력 데이터에 대해 batch normalization 수행
                            # num_features를 입력으로 받아 배치 정규화 수행

# Layer normalization
print(nn.LayerNorm(3, 3)(x)) # 정규화하려는 차원 크기로 계층 정규화 수행

# Instance normalization
print(nn.InstanceNorm1d(3)(x)) # 2D/3D 입력 데이터에 대해 instance normalization 수행


