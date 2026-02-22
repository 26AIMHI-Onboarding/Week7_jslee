import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 1. 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 2. 네트워크 생성
# [빈칸 1] 입력층 노드 수 설정
# Hint: MNIST 이미지는 28 x 28 픽셀입니다. 이를 1줄로 펴면 몇 개일까요?
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 3. 하이퍼파라미터 설정
iters_num = 10000  # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# [빈칸 2] 1에폭(Epoch)당 반복 수 계산
# Hint: 전체 데이터(train_size)를 미니배치(batch_size)로 나누면 1에폭에 필요한 반복 횟수가 나옵니다.
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # [빈칸 3] 미니배치 획득
    # Hint: 전체 데이터 인덱스 중 'batch_size' 개수만큼 랜덤하게 뽑습니다.
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    grad = network.gradient(x_batch, t_batch)
    
    # [빈칸 4] 매개변수 갱신 (SGD)
    # Hint: 가중치를 업데이트할 때 '학습률(learning_rate)'을 곱해서 빼줍니다.
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # [빈칸 5] 1에폭마다 정확도 계산
    # Hint: 반복 횟수(i)가 1에폭 주기(iter_per_epoch)로 나누어 떨어질 때마다 로그를 출력합니다.
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()