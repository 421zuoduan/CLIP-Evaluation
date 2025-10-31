import os
import requests
from pathlib import Path

# 设置 Hugging Face 镜像端点
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
BASE_URL = f"{HF_ENDPOINT}/datasets/timm/resisc45/resolve/main"

# 目标目录
TARGET_DIR = Path("./data/resisc45")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# 要下载的文件列表
FILES_TO_DOWNLOAD = [
    "README.md",
    "data/test-00000-of-00001.parquet"
]

def download_file(url: str, target_path: Path) -> bool:
    """下载单个文件"""
    try:
        print(f"正在下载: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ 下载完成: {target_path}")
        return True
    except Exception as e:
        print(f"❌ 下载失败 {url}: {str(e)}")
        return False

def main():
    """主下载函数"""
    print(f"使用镜像端点: {HF_ENDPOINT}")
    print(f"目标目录: {TARGET_DIR}")
    
    success_count = 0
    total_files = len(FILES_TO_DOWNLOAD)
    
    for file_path in FILES_TO_DOWNLOAD:
        url = f"{BASE_URL}/{file_path}"
        target_path = TARGET_DIR / file_path
        
        # 如果文件已存在，跳过下载
        if target_path.exists():
            print(f"⏭️  文件已存在，跳过: {target_path}")
            success_count += 1
            continue
        
        if download_file(url, target_path):
            success_count += 1
    
    print(f"\n📊 下载统计: {success_count}/{total_files} 文件成功")
    
    if success_count == total_files:
        print("🎉 RESISC45 数据集下载完成！")
        print(f"📁 数据保存位置: {TARGET_DIR}")
    else:
        print("⚠️  部分文件下载失败，请检查错误信息")

if __name__ == "__main__":
    main()