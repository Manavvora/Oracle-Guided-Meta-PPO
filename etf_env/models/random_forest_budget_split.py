# 文件名: random_forest_budget_split.py
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# 假设您的BudgetSplitModel类保存在这个路径
from models.random_forest import BudgetSplitModel

def load_training_data(data_dir: str):
    path_X = os.path.join(data_dir, "features_X.npy")
    path_y = os.path.join(data_dir, "target_y.npy")
    if not os.path.exists(path_X) or not os.path.exists(path_y):
        raise FileNotFoundError(f"在目录中找不到训练数据: {data_dir}")
    X = np.load(path_X)
    y = np.load(path_y)
    return X, y

def train_rf_model(data_dir: str, output_model_path: str):
    """
    执行完整的随机森林模型训练流程。
    """
    X, y = load_training_data(data_dir)
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Data split into: {len(X_train)} training and {len(X_test)} test samples.")

    # [最终修正] BudgetSplitModel的初始化不再需要 use_hyperparam_search 参数
    # 因为这个逻辑已经被移入 .train() 方法中
    model = BudgetSplitModel(random_state=42)

    print("\n--- Starting Random Forest Model Training ---")
    # 我们将测试集传入.train()方法，它会在训练后自动打印性能
    # use_hyperparam_search=True 这个开关现在是 .train() 方法的参数
    model.train(X_train, y_train, X_test, y_test, use_hyperparam_search=True)
    
    model.save(output_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a Random Forest model for budget allocation."
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True,
        help='Path to the directory containing features_X.npy and target_y.npy'
    )
    parser.add_argument(
        '--output_model', 
        type=str, 
        required=True,
        help='Path to save the trained model (e.g., results/rf_model.joblib)'
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    train_rf_model(args.data_dir, args.output_model)