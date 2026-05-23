import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

def run_ultimate_classic_svm():
    print("=== 1. 数据准备 (非线性 Moon 数据集 + 噪声) ===")
    # 按照建议：使用非线性数据集 make_moons，并加入 0.2 的噪声，让分类变难
    X, y = make_moons(n_samples=150, noise=0.2, random_state=42)
    
    # 按照建议：固定随机种子，保证每次运行结果一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 按照建议：数据标准化 (极其重要！)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("数据加载完成！训练集:", X_train.shape, "测试集:", X_test.shape)
    
    # ==========================================
    # 模型 1：传统 SVM (线性核 - 注定失败的对照组)
    # ==========================================
    print("\n=== 2. 传统 SVM (Linear 核) ===")
    svm_linear = SVC(kernel='linear', probability=True, random_state=42)
    
    # 交叉验证
    cv_scores_linear = cross_val_score(svm_linear, X_train, y_train, cv=5)
    print(f"5折交叉验证平均分: {cv_scores_linear.mean()*100:.2f}%")
    
    start_time = time.time()
    svm_linear.fit(X_train, y_train)
    time_linear = time.time() - start_time
    
    y_pred_linear = svm_linear.predict(X_test)
    acc_linear = accuracy_score(y_test, y_pred_linear)
    print(f"测试集准确率: {acc_linear * 100:.2f}%")
    print(f"训练耗时: {time_linear:.4f} 秒")

    # ==========================================
    # 模型 2：传统 SVM (RBF 核 - 高阶对照组)
    # ==========================================
    print("\n=== 3. 传统 SVM (RBF 核) ===")
    # 打印参数以备报告使用
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    print(f"模型参数: {svm_rbf.get_params()}")
    
    # 交叉验证
    cv_scores_rbf = cross_val_score(svm_rbf, X_train, y_train, cv=5)
    print(f"5折交叉验证平均分: {cv_scores_rbf.mean()*100:.2f}%")
    
    start_time = time.time()
    svm_rbf.fit(X_train, y_train)
    time_rbf = time.time() - start_time
    
    y_pred_rbf = svm_rbf.predict(X_test)
    acc_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f"测试集准确率: {acc_rbf * 100:.2f}%")
    print(f"训练耗时: {time_rbf:.4f} 秒")
    
    print("\n=== 分类详细报告 (RBF核) ===")
    print(classification_report(y_test, y_pred_rbf))
    
    # ==========================================
    # 可视化输出：混淆矩阵与 ROC 曲线
    # ==========================================
    print("\n=== 正在生成图表 (关闭图表窗口后程序结束) ===")
    
    # 创建一个 1x2 的画板
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 画混淆矩阵 (RBF 核)
    cm = confusion_matrix(y_test, y_pred_rbf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax1)
    ax1.set_title("Confusion Matrix (RBF Kernel)")
    
    # 画 ROC 曲线 (对比 Linear 和 RBF)
    RocCurveDisplay.from_estimator(svm_linear, X_test, y_test, ax=ax2, name="Linear SVM")
    RocCurveDisplay.from_estimator(svm_rbf, X_test, y_test, ax=ax2, name="RBF SVM")
    ax2.set_title("ROC Curve Comparison")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray") # 对角线
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_ultimate_classic_svm()

    