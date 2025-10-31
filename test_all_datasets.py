#!/usr/bin/env python3
"""
测试CLIP ViT-L/14在多个数据集上的性能，并将结果保存到Excel文件

该脚本会测试除了country211、imagenetv2、sun397、objectnet和eurosat外的所有支持数据集，
不限制样本数量，并将结果保存到Excel文件中。
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from main import (
    DATASET_CLASSES,
    evaluate_dataset,
    load_clip_model,
    setup_device,
    setup_dtype,
    create_directories
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="测试CLIP ViT-L/14在多个数据集上的性能",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="数据集根目录"
    )
    
    parser.add_argument(
        "--model-root",
        type=str,
        default="./model_weights",
        help="CLIP 模型根目录"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="结果输出目录"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="数据加载器的工作进程数"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备 (auto/cuda/cpu)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="数据类型 (auto/fp16/fp32)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="计算 top-k 准确率"
    )
    
    parser.add_argument(
        "--excel-name",
        type=str,
        default=None,
        help="Excel文件名（不包含扩展名），默认使用时间戳"
    )
    
    return parser.parse_args()


def save_results_to_excel(results, output_dir, excel_name=None):
    """将评估结果保存到Excel文件"""
    if excel_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_name = f"clip_evaluation_{timestamp}"
    
    excel_path = os.path.join(output_dir, f"{excel_name}.xlsx")
    
    # 准备数据
    data = []
    for result in results:
        if "error" in result:
            # 处理失败的情况
            data.append({
                "数据集": result["dataset"],
                "样本数量": "N/A",
                "类别数量": "N/A",
                "Top-1 准确率": "N/A",
                f"Top-{result.get('top_k', 5)} 准确率": "N/A",
                "推理时间(秒)": "N/A",
                "处理速度(样本/秒)": "N/A",
                "状态": "失败",
                "错误信息": result["error"]
            })
        else:
            # 处理成功的情况
            top_k = result.get('top_k', 5)
            data.append({
                "数据集": result["dataset"],
                "样本数量": result["num_samples"],
                "类别数量": result["num_classes"],
                "Top-1 准确率": f"{result['top1_accuracy']:.4f}",
                f"Top-{top_k} 准确率": f"{result[f'top{top_k}_accuracy']:.4f}",
                "推理时间(秒)": f"{result['inference_time']:.2f}",
                "处理速度(样本/秒)": f"{result['samples_per_second']:.2f}",
                "状态": "成功",
                "错误信息": ""
            })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存到Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='评估结果', index=False)
        
        # 获取工作表对象
        worksheet = writer.sheets['评估结果']
        
        # 调整列宽
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"\n💾 结果已保存到Excel文件: {excel_path}")
    
    # 打印摘要
    successful = len([r for r in results if "error" not in r])
    failed = len([r for r in results if "error" in r])
    
    print("\n📋 评估摘要:")
    print(f"总数据集数: {len(results)}")
    print(f"成功评估: {successful}")
    print(f"失败评估: {failed}")
    
    if successful > 0:
        print("\n🏆 最佳结果 (Top-1 准确率):")
        successful_results = [r for r in results if "error" not in r]
        successful_results.sort(key=lambda x: x.get("top1_accuracy", 0), reverse=True)
        
        for i, result in enumerate(successful_results[:5]):
            print(f"{i+1}. {result['dataset']}: {result.get('top1_accuracy', 0):.4f}")
    
    return excel_path


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = setup_device(args.device)
    
    # 设置数据类型
    dtype = setup_dtype(args.dtype, device)
    
    # 创建输出目录
    create_directories(args.output_dir)
    
    # 获取所有支持的数据集，排除country211、imagenetv2、sun397、objectnet和eurosat
    all_datasets = list(DATASET_CLASSES.keys())
    exclude_datasets = ["country211", "imagenet_v2", "sun397", "objectnet", "eurosat"]
    dataset_names = [d for d in all_datasets if d not in exclude_datasets]
    
    print(f"🎯 将评估以下数据集 (排除 {exclude_datasets}): {', '.join(dataset_names)}")
    
    # 加载 CLIP 模型
    print("\n🤖 加载 CLIP 模型...")
    try:
        model, processor = load_clip_model(args.model_root, device, dtype)
    except Exception as e:
        print(f"❌ 加载 CLIP 模型失败: {str(e)}")
        return
    
    # 评估所有数据集
    results = []
    start_time = time.time()
    
    for i, dataset_name in enumerate(dataset_names):
        print(f"\n📊 [{i+1}/{len(dataset_names)}] 开始评估数据集: {dataset_name}")
        
        result = evaluate_dataset(
            model=model,
            processor=processor,
            dataset_name=dataset_name,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            limit=None,  # 不限制样本数量
            top_k=args.top_k
        )
        result['top_k'] = args.top_k  # 保存top_k值
        results.append(result)
    
    total_time = time.time() - start_time
    print(f"\n⏱️  总评估时间: {total_time:.2f} 秒")
    
    # 保存结果到Excel
    excel_path = save_results_to_excel(results, args.output_dir, args.excel_name)
    
    print("\n🎉 评估完成!")
    return excel_path


if __name__ == "__main__":
    main()