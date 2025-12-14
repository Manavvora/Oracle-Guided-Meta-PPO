import os
import argparse
import joblib
import numpy as np
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class BudgetSplitModel:
    """
    一个健壮的随机森林模型类，用于根据组件特征预测其预算分配比例。
    """
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model: Optional[RandomForestRegressor] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.param_distributions = {
            "n_estimators": [100, 300, 500, 800], "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None, use_hyperparam_search: bool = True, n_iter_search: int = 20):
        if use_hyperparam_search:
            from sklearn.model_selection import RandomizedSearchCV
            base_rf = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
            print("▶ 启动 RandomizedSearchCV 进行超参数调优...")
            search = RandomizedSearchCV(estimator=base_rf, param_distributions=self.param_distributions, n_iter=n_iter_search, cv=5, scoring="neg_mean_squared_error", random_state=self.random_state, n_jobs=self.n_jobs, verbose=1)
            search.fit(X_train, y_train)
            self.best_params_ = search.best_params_
            print(f"🔍 找到的最佳超参数: {self.best_params_}")
            self.model = search.best_estimator_
        else:
            print("▶ 使用默认参数进行训练...")
            self.model = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
            self.model.fit(X_train, y_train)

        if X_test is not None and y_test is not None:
            self.evaluate(X_test, y_test)
        print("--- 模型训练完成 ---")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        if self.model is None: raise ValueError("模型尚未训练，无法评估。")
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"✔ 在测试集上的性能: MSE = {mse:.6f}, R² = {r2:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None: raise ValueError("模型尚未训练，请先调用 .train() 方法。")
        return self.model.predict(X)

    def save(self, path: str):
        if self.model is None: raise ValueError("没有可供保存的模型。")
        dir_path = os.path.dirname(path)
        if dir_path: os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self, path)
        print(f"✅ 模型对象已保存至: {path}")

    # [最终修正] 将 load 方法修改为类方法
    @classmethod
    def load(cls, path: str) -> 'BudgetSplitModel':
        """
        从文件加载一个BudgetSplitModel的实例。这是一个类方法。
        """
        try:
            model_instance = joblib.load(path)
            print(f"✅ 从 {path} 成功加载模型对象。")
            return model_instance
        except FileNotFoundError:
            raise FileNotFoundError(f"错误: 找不到模型文件 {path}。")

def main():
    parser = argparse.ArgumentParser(description="训练一个用于预算分配的随机森林模型。")
    parser.add_argument('--data_dir', type=str, required=True, help='包含 features_X.npy 和 target_y.npy 的目录路径。')
    parser.add_argument('--output_model', type=str, required=True, help='保存训练好的模型对象的文件路径。')
    args = parser.parse_args()

    X_path = os.path.join(args.data_dir, "features_X.npy")
    y_path = os.path.join(args.data_dir, "target_y.npy")
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"加载数据成功: X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"数据已划分为: {len(X_train)} 个训练样本和 {len(X_test)} 个测试样本。")

    model = BudgetSplitModel()
    model.train(X_train, y_train, X_test, y_test)
    model.save(args.output_model)

if __name__ == "__main__":
    main()