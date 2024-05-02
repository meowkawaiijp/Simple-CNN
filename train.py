import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from network.CNN import CNN 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
epoch_num = 5 #エポック数
transform = transforms.Compose([
    transforms.ToTensor(),  # 画像をテンソルに変換
    transforms.Normalize((0.5,), (0.5,))  # 平均、偏差を0.5で正規化
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)  # training用のデータセットを用意

def train():
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # オプティマイザーはAdam
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習
    for epoch in tqdm(range(epoch_num), desc='Training'):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()  # 勾配を初期化
            outputs = model(inputs)  # モデルにデータを渡して予測を取得
            loss = criterion(outputs, labels)  # 損失を計算
            loss.backward()  # 逆伝播
            optimizer.step()  # パラメータを更新
            running_loss += loss.item()
            # if i % 10 == 0:
            #     print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/(i+1):.4f}')
    # print(f'Total Batches: {len(trainloader)}')
    print('Finished Training')
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')

if __name__ == "__main__":
    train()
