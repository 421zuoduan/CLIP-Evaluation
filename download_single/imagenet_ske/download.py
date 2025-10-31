import os
from pathlib import Path

# 设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"❌ 缺少 huggingface_hub 库: {e}")
    print("💡 请运行: pip install huggingface_hub")
    exit(1)

# 配置下载参数
repo_id = "songweig/imagenet_sketch"
local_dir = "./data/imagenet_ske"
cache_dir = "./data/imagenet_ske/__hf_cache__"

# 确保目录存在
Path(local_dir).mkdir(parents=True, exist_ok=True)
Path(cache_dir).mkdir(parents=True, exist_ok=True)

print(f"🔄 开始从 {repo_id} 下载 ImageNet-Sketch 数据集...")
print(f"📁 保存路径: {local_dir}")
print(f"🌐 使用镜像: {os.environ['HF_ENDPOINT']}")

try:
    # 从 Hugging Face Hub 下载数据集
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    
    print("✅ ImageNetV2 数据集下载完成！")
    print(f"📂 数据保存至: {downloaded_path}")
    
    # 列出下载的内容
    print("\n📋 下载的内容:")
    for item in Path(local_dir).iterdir():
        if item.is_file():
            print(f"  📄 {item.name}")
        elif item.is_dir():
            file_count = len(list(item.iterdir()))
            print(f"  📁 {item.name}/ (包含 {file_count} 个文件/子目录)")
            
except Exception as e:
    print(f"❌ 下载失败: {str(e)}")
    print("💡 请检查网络连接或确认路径是否有写入权限")