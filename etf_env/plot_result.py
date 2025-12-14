#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_final_results(
    results_dir: str,
    output_path: str
):

    print("--- ploting ---")

    # --- 设置绘图风格 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 定义模型名称和对应的文件名
    model_files = {
        'baseline': 'baseline_survival_times.npy',
        'Vanilla PPO': 'vanilla_ppo_survival_times.npy',
        'Meta-PPO (Oracle-Guided)': 'meta_ppo_survival_times.npy',
        'Oracle': 'oracle_survival_times.npy'
    }

    results_data = []

    for model_name, file_name in model_files.items():
        file_path = os.path.join(results_dir, file_name)
        
        if os.path.exists(file_path):
            survival_times = np.load(file_path)
            for time in survival_times:
                results_data.append({'Strategy': model_name, 'Survival Time': time})
            print(f"✅ successful: {model_name}")
        else:
            print(f"⚠️{file_path}")

    if not results_data:
        print("❌ ")
        return

    df = pd.DataFrame(results_data)

    sns.barplot(
        data=df, 
        x='Strategy', 
        y='Survival Time',
        palette='viridis',
        capsize=.1 
    )

    plt.title('Comparison of Average Survival Time by Strategy', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Strategy', fontsize=16)
    plt.ylabel('Average Survival Time (Days)', fontsize=16)
    plt.xticks(rotation=15, ha="right", fontsize=14)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # --- 保存图片 ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot final result comparison for different models.")
    parser.add_argument("--results_dir", type=str, required=True, help="")
    parser.add_argument("--output_path", type=str, default="results/final_comparison.png", help="save")
    args = parser.parse_args()
    
    plot_final_results(
        results_dir=args.results_dir,
        output_path=args.output_path
    )