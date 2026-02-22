import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 1. 데이터 준비 및 학습 (간단하게 1000회만 수행)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 1000 
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

print("학습 중입니다... 잠시만 기다려주세요.")
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.gradient(x_batch, t_batch)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

print("학습 완료! 가중치(W1)를 시각화합니다.")

# W1 가중치 가져오기
# W1의 형상(Shape)은 (784, 50) 입니다. (입력 784개 -> 은닉 50개)
W1 = network.params['W1']

plt.figure(figsize=(10, 5))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# 앞에서부터 20개의 은닉 뉴런이 무엇을 보고 있는지 그립니다.
for i in range(20):
    plt.subplot(4, 5, i + 1) # 4행 5열로 배치
    
    # [빈칸 1] i번째 은닉 뉴런의 가중치 추출
    # Hint: W1은 2차원 배열입니다. 모든 행(:)의 i번째 열을 가져와야 합니다.
    weight = W1[:, i]
    
    # [빈칸 2] 1줄로 되어있는 가중치를 이미지 모양(28x28)으로 변형
    # Hint: 입력 데이터가 원래 28x28 픽셀이었음을 기억하세요. 
    weight_img = weight.reshape(28, 28)
    
    plt.imshow(weight_img, cmap='gray')
    
    plt.axis('off') # 축 눈금 제거

plt.show()