import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 1. 데이터 준비 및 네트워크 생성
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 입력 784(이미지 크기) -> 은닉 50 -> 출력 10
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 2. 하이퍼파라미터 설정
iters_num = 2000  # 시각화를 위해 적당히 학습 (2000회 정도면 패턴이 보임)
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

print("학습을 시작합니다... (잠시만 기다려주세요)")

# 3. 학습 루프
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 계층(Layer) 버전의 gradient는 역전파를 통해 효율적으로 계산됩니다.
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    if i % 500 == 0:
        print(f"반복 {i} | 손실: {network.loss(x_batch, t_batch):.4f}")

print("학습 완료! 가중치 시각화를 진행합니다.")

# [시각화 실습] 학습된 1층 가중치(W1) 그려보기

# 가중치 W1 가져오기
# W1의 형상(Shape): (784, 50) -> (입력 픽셀 수, 은닉 뉴런 수)
weights = network.params['W1']

plt.figure(figsize=(10, 5))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

print("상위 20개 필터(뉴런) 시각화 중...")

# 20개의 은닉 뉴런(필터)을 시각화
for i in range(20):
    plt.subplot(4, 5, i + 1)
    
    # [빈칸 1] i번째 은닉 뉴런에 해당하는 가중치 추출
    # Hint: weights는 (784, 50) 2차원 배열입니다. 모든 행(:)의 i번째 열을 가져오세요.
    w = weights[:, i]
    
    # [빈칸 2] 1차원 배열(784개)을 2차원 이미지(28x28)로 변환
    # Hint: 그림을 그리려면 가로x세로 형태여야 합니다. numpy의 reshape 함수를 사용하세요.
    w_img = w.reshape(28, 28)
    
    plt.imshow(w_img, cmap='gray')
    
    plt.axis('off') # 축 눈금 제거

plt.show()