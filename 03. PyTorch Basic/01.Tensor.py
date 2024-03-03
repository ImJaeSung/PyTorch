import torch
import numpy as np

# 텐서 생성
print(torch.tensor([1, 2, 3])) # 입력된 데이터를 복사해 텐서로 변환하는 함수, Int 형식으로 할당
                               # 값이 무조건 존재해야함
print(torch.Tensor([[1, 2, 3], [4, 5, 6]])) # 텐서 인스턴스를 생성하는 클래스, Float 형식으로 할당

## torch.Tensor클래스를 상속받은 형식으로 미리 데이터 형식이 선언된 클래스
print(torch.LongTensor([1, 2, 3])) 
print(torch.FloatTensor([1, 2, 3]))
print(torch.IntTensor([1, 2, 3]))

# 텐서 속성
tensor = torch.rand(1, 2) # 0과 1사이에서 무작위 숫자를 균등 분포로 생성하는 함수
print(tensor)
print(tensor.shape)
print(tensor.dtype) 
print(tensor.device)

# 차원 변환
tensor = torch.rand(1, 2)
print(tensor)
print(tensor.shape)

tensor = tensor.reshape(2, 1)
print(tensor)
print(tensor.shape)

# 텐서 자료형 설정
tensor = torch.rand((3, 3), dtype = torch.float) # 32bit 부동 소수점
tensor = torch.rand((3, 3), dtype = float) # 64bit 부동 소수점
print(tensor)

# 장치 설정
cpu = torch.FloatTensor([1, 2, 3])
gpu = torch.cuda.FloatTensor([1, 2, 3])
print(cpu)
print(gpu)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu" # mac ver.
tensor = torch.rand((1, 1), device = device)
print(device)
print(tensor)

# 장치 변환
## CPU 장치를 사용하는 텐서와 GPU 장치를 사용하는 텐서는 상호 간 연산이 불가능
## CPU 장치를 사용하는 텐서와 넘파이 배열 간 연산은 가능
## GPU 장치를 사용하는 텐서와 넘파이 배열 간 연산은 불가능
cpu = torch.FloatTensor([1, 2, 3])
gpu = cpu.cuda()
gpu2cpu = gpu.cpu()
cpu2gpu = cpu.to("cuda") # to 메서드로 장치를 간단하게 변환가능
print(cpu)
print(gpu)
print(gpu2cpu)
print(cpu2gpu)

# 넘파이 배열의 텐서 변환
ndarray = np.array([1, 2, 3], dtype = np.uint8)
print(torch.tensor(ndarray))
print(torch.Tensor(ndarray)
print(torch.from_numpy(ndarray))

# 텐서의 넘파이 배열 변환
tensor = torch.cuda.FloatTensor([1, 2, 3])
ndarray = tensor.detach().cpu().numpy() # 텐서는 학습을 위한 데이터 형식으로 모든 연산을 추적해  기록하여 역전파 등과 같은 연산이 진행돼 모델 학습이 이루어짐
                                        # detach 메서드는 현재 연산 그래프에서 분리된 새로운 텐서를 반환
print(ndarray)
print(dtype(ndarray))