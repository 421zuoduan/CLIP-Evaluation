import torchvision
import torchvision.transforms as transforms

# 数据预处理（转Tensor并归一化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载 CIFAR-10
# trainset_cifar10 = torchvision.datasets.CIFAR10(
#     root='./data/cifar10',
#     train=True,
#     download=True,
#     transform=transform
# )
testset_cifar10 = torchvision.datasets.CIFAR10(
    root='./data/cifar10',
    train=False,
    download=True,
    transform=transform
)

