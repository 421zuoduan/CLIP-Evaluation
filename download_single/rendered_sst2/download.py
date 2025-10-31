# 1. 导入必要库（需确保 torch 和 torchvision 已安装）
from torchvision.datasets import RenderedSST2

# 2. 配置数据集参数（核心：指定 split="test" 仅下载测试集）
dataset_params = {
    "root": "./data/rendered_sst2",  # 测试集保存路径（不存在会自动创建）
    "split": "test",                 # 关键参数：仅下载测试集
    "download": True,                # 触发下载（首次运行设为 True，后续设为 False）
    "transform": None,               # 无需预处理（仅下载时可设为 None）
    "target_transform": None,        # 标签无需预处理（仅下载时可设为 None）
    "loader": None                   # 使用默认图像加载器（不影响下载）
}

# 3. 创建数据集实例（执行此步骤即开始下载测试集）
try:
    rendered_sst2_test = RenderedSST2(**dataset_params)
    print("✅ RenderedSST2 测试集下载完成！")
    print(f"📊 测试集包含 {len(rendered_sst2_test)} 张图像（909张积极 / 912张消极）")
    print(f"💾 测试集保存路径：{dataset_params['root']}")
except Exception as e:
    print(f"❌ 下载失败：{str(e)}")
    print("💡 请检查网络连接或确认 root 路径是否有写入权限")