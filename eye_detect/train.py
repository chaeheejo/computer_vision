import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from dataclass import eyes_dataset
from model import Map
import torch.optim as optim

torch.set_num_threads(1)

#numpy 형태로 저장한 train dataset을 활용
x_train = np.load('./dataset/x_train.npy').astype(np.float32)  # (2586, 26, 34, 1)
y_train = np.load('./dataset/y_train.npy').astype(np.float32)  # (2586, 1)

#코랩 환경에서 cuda 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
])

train_dataset = eyes_dataset(x_train, y_train, transform=train_transform)

#accuracy 값 측정
def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

PATH = 'weights/classifier_weights_iter_50.pt'

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

#모델 생성
model = Map()
model.to(device)

#손실함수는 binary-cross-entropy 사용, 옵티마이저는 Adam을 사용
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#epoch 설정, 30으로 설정해도 학습이 충분히 되는 것을 확인
epochs = 50

#학습 진행
for epoch in range(epochs):
    running_loss = 0.0
    running_acc = 0.0

    model.train()

    for i, data in enumerate(train_dataloader, 0):
        input_1, labels = data[0].to(device), data[1].to(device)

        input = input_1.transpose(1, 3).transpose(2, 3)
        optimizer.zero_grad()

        outputs = model(input)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)

        if i % 80 == 79:
            print('epoch: [%d/%d] train_loss: %.5f train_acc: %.5f' % (
                epoch + 1, epochs, running_loss / 80, running_acc / 80))
            running_loss = 0.0

print("learning finish")
torch.save(model.state_dict(), PATH)