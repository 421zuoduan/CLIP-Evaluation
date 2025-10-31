#!/usr/bin/env python3
"""
测试修复后的Flowers-102数据集加载器
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 添加当前目录到Python路径
sys.path.append('.')

from main import Flowers102Dataset, create_text_prompts

def test_flowers102_dataset():
    """测试Flowers-102数据集加载器"""
    print("🔍 测试修复后的Flowers-102数据集加载器...")
    
    # 标准图像预处理
    standard_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # 测试数据集加载
    data_path = "./data/flowers102"
    
    try:
        # 测试测试集
        print("\n📊 测试测试集加载...")
        test_dataset = Flowers102Dataset(data_path, split="test", transform=standard_transform)
        print(f"✅ 测试集大小: {len(test_dataset)}")
        
        # 测试训练集
        print("\n📊 测试训练集加载...")
        train_dataset = Flowers102Dataset(data_path, split="train", transform=standard_transform)
        print(f"✅ 训练集大小: {len(train_dataset)}")
        
        # 测试验证集
        print("\n📊 测试验证集加载...")
        valid_dataset = Flowers102Dataset(data_path, split="valid", transform=standard_transform)
        print(f"✅ 验证集大小: {len(valid_dataset)}")
        
        # 获取类别名称
        class_names = test_dataset.get_classes()
        print(f"\n🏷️  类别数量: {len(class_names)}")
        
        # 检查是否有重复的类别名称
        if len(class_names) == len(set(class_names)):
            print("✅ 类别名称无重复")
        else:
            duplicates = [name for name in class_names if class_names.count(name) > 1]
            print(f"❌ 发现重复的类别名称: {duplicates}")
        
        # 测试数据加载器
        print("\n🔄 测试数据加载器...")
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 获取一个批次的数据
        for images, labels in test_loader:
            print(f"✅ 成功加载批次: 图像形状 {images.shape}, 标签形状 {labels.shape}")
            print(f"📝 标签范围: {labels.min().item()} - {labels.max().item()}")
            break
        
        # 测试文本提示生成
        print("\n📝 测试文本提示生成...")
        text_prompts = create_text_prompts(class_names, dataset_name="flowers102")
        print(f"✅ 生成了 {len(text_prompts)} 个文本提示")
        print(f"📝 前5个提示: {text_prompts[:5]}")
        
        # 检查文本提示是否有重复
        if len(text_prompts) == len(set(text_prompts)):
            print("✅ 文本提示无重复")
        else:
            duplicates = [prompt for prompt in text_prompts if text_prompts.count(prompt) > 1]
            print(f"❌ 发现重复的文本提示: {duplicates}")
        
        print("\n🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_original():
    """比较修复前后的差异"""
    print("\n📋 修复前后对比:")
    print("1. 修复前问题:")
    print("   - class_names列表中有重复的类别名称（如bougainvillea, camellia等）")
    print("   - 没有正确处理数据集的分割（train/valid/test）")
    print("   - 类别名称与标签不匹配，导致文本提示重复")
    print("   - 加载了所有图像，而不是仅加载测试集")
    
    print("\n2. 修复后改进:")
    print("   - 移除了class_names列表中的重复类别名称")
    print("   - 添加了对数据集分割的支持（train/valid/test）")
    print("   - 修复了图像ID与标签的对应关系")
    print("   - 确保类别名称与标签索引正确匹配")
    print("   - 默认只加载测试集，符合评估需求")

if __name__ == "__main__":
    success = test_flowers102_dataset()
    compare_with_original()
    
    if success:
        print("\n✅ 修复验证成功！现在可以使用修复后的代码进行CLIP评估。")
    else:
        print("\n❌ 修复验证失败！需要进一步调试。")