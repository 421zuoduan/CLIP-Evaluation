import torchvision
import torchvision.transforms as transforms

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 下载训练集（PyTorch 会自动下载并解压）
dataset = torchvision.datasets.SUN397(
    root='./data/sun397',
    download=True,
    transform=transform
)

print(len(dataset))      # ~108k
print(dataset[0])        # (image_tensor, label_index)
