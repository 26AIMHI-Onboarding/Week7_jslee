import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        # [빈칸 1] 입력 데이터(x)와 1층 가중치(W1)의 내적에 편향(b1)을 더함
        # Hint: numpy의 dot 함수 사용
        a1 = np.dot(x, W1) + b1
        
        # [빈칸 2] 1층의 출력을 활성화 함수(Sigmoid)에 통과시킴
        z1 = sigmoid(a1)
        
        # [빈칸 3] 1층의 출력(z1)과 2층 가중치(W2)의 내적에 편향(b2)을 더함
        a2 = np.dot(z1, W2)+b2
        
        # [빈칸 4] 최종 출력을 확률로 변환 (Softmax)
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        # [빈칸 5] 예측값(y)과 정답(t) 사이의 교차 엔트로피 오차 계산
        return cross_entropy_error(y,t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        
        # [빈칸 6] 확률(y)과 정답(t)에서 가장 값이 큰 인덱스(axis=1)를 가져옴
        # Hint: numpy의 argmax 함수 사용
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # 역전파(Backpropagation) 방법으로 기울기 구하기
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

# 실행부 (Main)
if __name__ == '__main__':
    # 1. 데이터 준비 (랜덤 데이터로 시뮬레이션)
    # 실제로는 MNIST 데이터를 로드하지만, 여기서는 동작 확인을 위해 랜덤값 사용
    input_size = 784
    hidden_size = 50
    output_size = 10
    
    x_train = np.random.randn(100, input_size) # 100개의 훈련 데이터
    t_train = np.zeros((100, output_size))     # 정답 레이블 (One-hot encoding)
    for i in range(100):
        t_train[i, np.random.randint(0, output_size)] = 1

    # 2. 네트워크 생성
    network = TwoLayerNet(input_size, hidden_size, output_size)

    # 3. 하이퍼파라미터 설정
    iters_num = 1000  # 반복 횟수
    train_size = x_train.shape[0]
    batch_size = 10   # 미니배치 크기
    learning_rate = 0.1

    print("학습 시작...")
    
    for i in range(iters_num):
        # 미니배치 획득
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # [빈칸 7] 기울기 계산
        # Hint: network의 gradient 메서드를 호출하여 기울기를 구하세요.
        grad = network.numerical_gradient(x_batch, t_batch)
        
        # [빈칸 8] 매개변수(가중치) 갱신 (SGD)
        # Hint: 학습률(learning_rate)과 기울기(grad)를 사용하여 params를 업데이트합니다.
        # W1, b1, W2, b2 모든 키에 대해 반복합니다.
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
            
        # 학습 경과 기록
        if i % 100 == 0:
            loss = network.loss(x_batch, t_batch)
            print(f"반복 {i} | 손실 함수 값: {loss:.4f}")

    print("학습 종료!")