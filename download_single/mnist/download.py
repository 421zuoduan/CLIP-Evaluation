import torchvision
import torchvision.transforms as transforms

# 定义预处理（转为Tensor并归一化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准均值和方差
])

# 下载测试集
testset = torchvision.datasets.MNIST(
    root='./data/mnist',
    train=False,
    download=True,
    transform=transform
)
