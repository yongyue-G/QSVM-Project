import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 数据生成模块 (内置于此，彻底解决导入报错)
# ==========================================
def load_nonlinear_data():
    """
    生成并预处理非线性的“同心月亮(Moons)”二分类数据集，加入噪声
    """
    X, y = make_moons(n_samples=150, noise=0.2, random_state=42)
    # 固定随机种子划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # 标准化 (极其重要)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# ==========================================
# 1. 基础模型训练与评估模块
# ==========================================
def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    print("\n=== 传统 SVM (Linear 核) ===")
    start_time_linear = time.time()
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)
    time_linear = time.time() - start_time_linear
    acc_linear = accuracy_score(y_test, y_pred_linear)
    print(f"准确率: {acc_linear * 100:.2f}%")
    print(f"运行时间: {time_linear:.4f} 秒\n")

    print("=== 传统 SVM (RBF 核) ===")
    start_time_rbf = time.time()
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)
    time_rbf = time.time() - start_time_rbf
    acc_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f"准确率: {acc_rbf * 100:.2f}%")
    print(f"运行时间: {time_rbf:.4f} 秒\n")

    print("=== 分类详细报告 (RBF核) ===")
    print(classification_report(y_test, y_pred_rbf))

    return svm_linear, svm_rbf

# ==========================================
# 2. 决策边界可视化模块
# ==========================================
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. 进阶硬核分析模块 (网格搜索、支持向量、核矩阵)
# ==========================================
def run_advanced_svm_analysis(X_train, X_test, y_train, y_test):
    print("\n=== 进阶分析 1: 穷举调参 (Grid Search) ===")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.1, 1, 'scale', 10]
    }
    grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print(f"找到的最优参数: {grid_search.best_params_}")
    print(f"该参数下的交叉验证最高分: {grid_search.best_score_ * 100:.2f}%")
    
    best_model = grid_search.best_estimator_
    print(f"测试集最终准确率: {best_model.score(X_test, y_test) * 100:.2f}%\n")

    print("=== 进阶分析 2: 支持向量深度剖析 ===")
    n_support = best_model.n_support_
    total_support = sum(n_support)
    print(f"训练集总样本数: {len(X_train)}")
    print(f"动用的支持向量总数: {total_support} 个 (占比 {total_support/len(X_train)*100:.1f}%)")
    print(f"类别 0 的支持向量数: {n_support[0]}")
    print(f"类别 1 的支持向量数: {n_support[1]}\n")

    print("=== 进阶分析 3: 提取并绘制核矩阵 (Kernel Matrix) ===")
    gamma_val = best_model.gamma if isinstance(best_model.gamma, float) else 1.0 / X_train.shape[1]
    K_matrix = rbf_kernel(X_train, X_train, gamma=gamma_val)
    
    sort_idx = np.argsort(y_train)
    K_matrix_sorted = K_matrix[sort_idx][:, sort_idx]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(K_matrix_sorted, cmap='viridis', origin='lower')
    plt.colorbar(label='Kernel Value (Similarity)')
    plt.title('Classic RBF Kernel Matrix (Gram Matrix)')
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('Sorted Sample Index')
    
    split_point = sum(y_train == 0)
    plt.axhline(split_point - 0.5, color='red', linestyle='--', alpha=0.7)
    plt.axvline(split_point - 0.5, color='red', linestyle='--', alpha=0.7)
    plt.text(split_point/2, split_point/2, 'Class 0 vs Class 0', color='white', ha='center')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序运行入口
# ==========================================
if __name__ == "__main__":
    # 1. 自动生成并加载数据
    X_train, X_test, y_train, y_test = load_nonlinear_data()

    # 2. 运行基础模型
    svm_linear, svm_rbf = train_and_evaluate_svm(X_train, X_test, y_train, y_test)

    # 3. 绘制并展示决策边界图
    print(">>> 正在绘制 Linear 核边界图，请关闭图片窗口以继续...")
    plot_decision_boundary(svm_linear, X_train, y_train, "Classic SVM (Linear Kernel)")
    
    print(">>> 正在绘制 RBF 核边界图，请关闭图片窗口以继续...")
    plot_decision_boundary(svm_rbf, X_train, y_train, "Classic SVM (RBF Kernel)")

    # 4. 运行硬核进阶分析并画出核矩阵热力图
    print(">>> 正在进行网格搜索与进阶分析...")
    run_advanced_svm_analysis(X_train, X_test, y_train, y_test)