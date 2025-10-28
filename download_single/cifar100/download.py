import torchvision
import torchvision.transforms as transforms

# 定义数据预处理：转Tensor并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),  # 均值
                         (0.2675, 0.2565, 0.2761))  # 方差
])

# 下载测试集
testset = torchvision.datasets.CIFAR100(
    root='./data/cifar100',
    train=False,
    download=True,
    transform=transform
)
