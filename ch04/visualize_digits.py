import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 1. 데이터 준비 및 학습
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 2000 # 형상이 뚜렷하게 보이도록 2000회 정도 학습
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

print("학습 중입니다... (약 10~20초 소요)")
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.gradient(x_batch, t_batch)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

print("학습 완료! 숫자 0부터 9까지의 학습된 형상을 시각화합니다.")

# 2. 가중치 가져오기
W1 = network.params['W1'] # (784, 50)
W2 = network.params['W2'] # (50, 10)

# [빈칸 1] 1층과 2층 가중치를 합쳐서(내적) 입력->출력의 관계를 하나로 만듦
# Hint: 두 행렬을 곱(dot)하면 (784, 10) 크기의 행렬이 됩니다.
combined_W = np.dot(W1, W2)

plt.figure(figsize=(12, 5))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 0부터 9까지 10개의 숫자에 대해 시각화
for i in range(10):
    plt.subplot(2, 5, i + 1) # 2행 5열로 배치
    
    # [빈칸 2] i번째 숫자에 해당하는 가중치 열(Column) 추출
    # Hint: combined_W의 i번째 열을 가져옵니다.
    weight = combined_W[:, i]
    
    # [빈칸 3] 1줄로 되어있는 가중치를 이미지 모양(28x28)으로 변형
    # Hint: numpy의 reshape 함수를 사용하세요.
    weight_img = weight.reshape(28, 28)
    
    # 'RdBu' 컬러맵을 사용하면 양수(파랑/빨강)와 음수(반대색) 대비가 잘 보입니다.
    plt.imshow(weight_img, cmap='RdBu_r') 
    
    plt.title(f"Label {i}") # 숫자 라벨 표시
    plt.axis('off')         # 축 눈금 제거

plt.show()