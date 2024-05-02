import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import sys

from network.CNN import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn_model.pth', map_location=device))  # 学習したモデルのパラメータを読み込み
transform = transforms.Compose([
    transforms.Grayscale(),   # グレースケール変換
    transforms.Resize((28, 28)),  # 画像サイズの変更
    transforms.ToTensor(),    # 画像をテンソルに変換
    transforms.Normalize((0.5,), (0.5,))  # 平均0.5、偏差0.5で正規化
])

def test(image_path=None, answer=None):
    global images, predictions, idx
    if image_path:
        image = Image.open(image_path) 
        image = transform(image).unsqueeze(0).to(device)
        images = image
    else:#テスト用の画像の指定がなければMNISTを使用する。
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)
        data_iter = iter(testloader)
        images, labels = next(data_iter)
        images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        idx = 0
        show_prediction(answer)

# 予測を表示
def show_prediction(answer=None):
    global idx
    prediction = predictions[idx].cpu().item()
    if answer:
        print("Answer:", answer)
    print("Prediction:", prediction)

if len(sys.argv) > 1:
    test(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None) 
else:
    test()