import torchvision
import torchvision.transforms as transforms

# 定义预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 下载训练集（PyTorch 会自动从 Caltech 官方源下载并解压）
dataset = torchvision.datasets.Caltech101(
    root='./data/caltech101',
    download=True,
    transform=transform
)

print(len(dataset))   # ~9144
print(dataset[0])     # (Tensor图像, 类别索引)
