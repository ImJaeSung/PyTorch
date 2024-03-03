import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# Simple regression : Numpy
x = np.array(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]
)
y = np.array(
    [[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]
)

weight = 100000.0
bias = 1000000.0
learning_rate = 0.001

for epoch in range(10000):
    y_hat = weight*x + bias
    cost = ((y - y_hat)**2).mean()

    weight -= learning_rate*((y_hat - y)*x).mean()
    bias -= learning_rate*(y_hat - y).mean()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Weight : {weight:.3f}, Bias : {bias:.3f}, Cost : {cost:.3f}")


# Simple regression : PyTorch
x = torch.FloatTensor(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]
)
y = torch.FloatTensor(
    [[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]
)

weight = torch.zeros(1, requires_grad = True) # 모든 텐서에 대한 연산을 추적하며 역전파 메서드를 호출해 기울기를 계산하고 저장, 자동미분 기능의 사용 여부
bias = torch.zeros(1, requires_grad = True) 
learning_rate = 0.001

optimizer = optim.SGD([weight, bias], lr = learning_rate)

for epoch in range(10000):
    hypothesis = x*weight + bias
    cost = torch.mean((hypothesis - y)**2)

    optimizer.zero_grad() # 텐서의 기울기는 grad 속성에 누적해서 더해지기 때문에 0으로 초기화
    cost.backward() # 역전파 수행하여 가중치와 편향에 대한 기울기 계산
    optimizer.step() # 최적화함수에 반영

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Weight : {weight.item():.3f}, Bias : {bias.item():.3f}, Cost : {cost:.3f}")


# .zero_grad(), .backward(), .step()
for epoch in range(3):
    hypothesis = weight*x + bias
    cost = torch.mean((hypothesis - y)**2)
    
    print(f"Epoch : {epoch+1:4d}")
    print(f"Step [1] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")

    optimizer.zero_grad()
    print(f"Step [2] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")

    cost.backward()
    print(f"Step [3] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")

    optimizer.step()
    print(f"Step [4] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")


# nn(Neural Network) package
model = nn.Linear(in_features = 1, out_features = 1, bias = True) # in_features, out_feature : weight, bias : bias
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr = learning_rate) # 모델 매개변수 전달

for epoch in range(10000):
    output = model(x)
    cost = criterion(output, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Weight : {weight.item():.3f}, Bias : {bias.item():.3f}, Cost : {cost:.3f}")
