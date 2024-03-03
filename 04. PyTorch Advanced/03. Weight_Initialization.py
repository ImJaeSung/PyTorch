import torch
import torch.nn as nn

# 상수 초기화 : (0, 1), 특정 상수, 단위행렬, 디랙 델타 함수(=임펄스 함수) 등
# Breaking Symmetry 

# 무작위 초기화 : Random, Uniform Distribution, Normal Distribution, Truncated Normal Distribution, Sparse Normal Distribution

# Xavier Initialization(=Clorot Initialization) : 입력 데이터의 분산이 출력 데이터에서 유지되도록 가중치 초기화
# 이전 계층의 노드 수, 다음 계층의 노드 수
# Sigmoid, Tanh

# Kaiming Initialization(=He Initialization) : 각 노드의 출력 분산이 입력 분산과 동일하도록 초기화
# 현재 계층의 입력 뉴런 수
# ReLU

# 직교 초기화 : SVD를 이용해 자기 자신을 제외한 나머지 모든 열, 행 벡터들과 직교이면서 동시에 단위 벡터인 행렬을 만드는 방법
# RNN 계열 모델에서 주로 사용

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 2),
            nn.Sigmoid()
        )
        
        self.fc = nn.Linear(2, 1)
        self._init_weights() # 모델 계층이 정의된 직후 호출

    def _init_weights(self): # Protected Method
        nn.init.xavier_uniform_(self.layer[0].weight) # Xavier Initialization
        self.layer[0].bias.data.fill_(0.01) # 상수 초기화

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)

model = Net()

# 모듈화하여 가중치 초기화 메서드 적용
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 2),
            nn.Sigmoid()
        )
        
        self.fc = nn.Linear(2, 1)
        self.apply(self._init_weights) # 텐서의 각 요소에 임의의 함수를 적용하고 결과와 함께 새 텐서를 반환

    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): # 객체 식별 함수로 선형 변환 함수인지 확인
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.01)

        print(f"Apply : {module}")

model = Net()