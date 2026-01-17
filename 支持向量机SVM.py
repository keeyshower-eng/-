import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def f_beta_score(precision, recall, beta=0.3):
    """计算F-beta分数"""
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

class TFIDF_SVM_SpamClassifier:
    """基于TF-IDF和支持向量机(SVM)的垃圾邮件分类器"""
    
    def __init__(self, random_state=RANDOM_STATE):
        """
        初始化分类器
        
        参数:
        random_state: 随机种子
        """
        self.random_state = random_state
        self.pipeline = None
        self.best_params_ = None
        self.best_score_ = None
        
    def load_and_prepare_data(self, filepath='spam mail.csv'):
        """
        加载和准备数据集
        """
        # 尝试不同的编码方式读取文件
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        if df is None:
            print("无法加载数据文件")
            return None
        
        # 检查并重命名列
        expected_columns = ['Category', 'Messages']
        for col in expected_columns:
            if col not in df.columns:
                for df_col in df.columns:
                    if col.lower() in df_col.lower():
                        df = df.rename(columns={df_col: col})
                        break
        
        # 确保列名正确
        if 'Category' not in df.columns or 'Messages' not in df.columns:
            print("数据集必须包含 'Category' 和 'Messages' 列")
            return None
        
        # 将类别转换为数值标签
        category_mapping = {'spam': 1, 'ham': 0, 'Spam': 1, 'Ham': 0}
        df['label'] = df['Category'].map(category_mapping)
        
        # 移除未映射的行
        if df['label'].isnull().any():
            df = df.dropna(subset=['label'])
        
        # 重命名列以便后续处理
        df = df.rename(columns={'Messages': 'text'})
        df['label'] = df['label'].astype(int)
        
        return df
    
    def preprocess_text(self, text):
        """
        文本预处理
        """
        if not isinstance(text, str):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 去除URL
        text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
        
        # 去除HTML标签
        text = re.sub(r'<.*?>', ' ', text)
        
        # 去除电子邮件地址
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # 去除电话号码
        text = re.sub(r'\b\d{10,}\b', ' PHONE ', text)
        
        # 去除特殊字符，保留基本标点
        text = re.sub(r'[^\w\s\.\?\!,]', ' ', text)
        
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self, df, test_size=0.2):
        """
        准备训练和测试数据
        """
        if df is None or len(df) == 0:
            return None, None, None, None
        
        # 应用文本预处理
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=df['label']
        )
        
        return X_train, X_test, y_train, y_test
    
    def create_pipeline(self):
        """
        创建TF-IDF + SVM的Pipeline
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svm', SVC(probability=True, random_state=self.random_state))
        ])
        
        return pipeline
    
    def perform_grid_search(self, X_train, y_train, cv=5):
        """
        执行网格搜索寻找最佳参数
        """
        # 定义参数网格
        param_grid = {
            'tfidf__max_features': [8000],
            'tfidf__ngram_range': [(1, 1)],
            'tfidf__stop_words': ['english'],
            'tfidf__min_df': [1],
            'tfidf__max_df': [0.8],
            'svm__C': [2],
            'svm__kernel': ['sigmoid'],
            'svm__gamma': ['scale'],
            'svm__class_weight': ['balanced']
        }
        
        # 创建Pipeline
        pipeline = self.create_pipeline()
        
        # 创建GridSearchCV对象
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring='f1',
            n_jobs=-1,
            verbose=0,
            return_train_score=True
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        # 保存最佳参数和分数
        self.pipeline = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        
        return grid_search
    
    def predict_with_threshold(self, X, threshold):
        """
        使用置信度阈值进行预测
        """
        # 获取预测概率
        proba = self.pipeline.predict_proba(X)
        
        # 获取垃圾邮件的概率
        spam_proba = proba[:, 1]
        
        # 只有当垃圾邮件概率大于阈值时才预测为垃圾邮件
        predictions = (spam_proba > threshold).astype(int)
        
        return predictions, spam_proba
    
    def evaluate_threshold(self, X_test, y_test, threshold):
        """
        评估特定阈值下的模型性能
        """
        # 使用阈值进行预测
        y_pred, y_pred_proba = self.predict_with_threshold(X_test, threshold)
        
        # 计算评估指标
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f03 = f_beta_score(precision, recall, beta=0.3)
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f03': f03
        }
    
    def find_optimal_threshold(self, X_test, y_test, threshold_range=None, num_points=50):
        """
        寻找使F0.3分数最大的最优置信阈值
        """
        if threshold_range is None:
            threshold_range = (0.1, 0.99)
        
        # 生成阈值序列
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        
        results = []
        best_f03 = -1
        best_result = None
        
        for threshold in thresholds:
            metrics = self.evaluate_threshold(X_test, y_test, threshold)
            results.append(metrics)
            
            if metrics['f03'] > best_f03:
                best_f03 = metrics['f03']
                best_result = metrics
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        return best_result, results_df
    
    def plot_metrics_vs_threshold(self, results_df, best_result):
        """
        绘制与图片中相似的简洁折线图
        包含两个子图：精确率/召回率/F1分数图和F0.3分数图
        """
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形，上下排列两个子图
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # 第一个子图：精确率/召回率/F1分数 vs 阈值
        ax1 = axes[0]
        
        # 绘制三条折线
        ax1.plot(results_df['threshold'], results_df['precision'], 
                label='精确率', linewidth=2.5, color='blue')
        ax1.plot(results_df['threshold'], results_df['recall'], 
                label='召回率', linewidth=2.5, color='red')
        
        # 标记最优阈值点（垂直虚线）
        ax1.axvline(x=best_result['threshold'], color='black', 
                   linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 设置坐标轴范围
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 添加标签和标题
        ax1.set_xlabel('置信阈值', fontsize=12)
        ax1.set_ylabel('指标值', fontsize=12)
        ax1.set_title('SVM 精确率/召回率', fontsize=14, fontweight='bold')
        
        # 添加图例
        ax1.legend(loc='lower left', fontsize=10)
        
        # 添加网格
        ax1.grid(True, alpha=0.3)
        
        # 第二个子图：F0.3分数 vs 阈值
        ax2 = axes[1]
        
        # 绘制F0.3分数折线
        ax2.plot(results_df['threshold'], results_df['f03'], 
                label='F0.3分数', linewidth=2.5, color='purple')
        
        # 标记最优阈值点（垂直虚线）
        ax2.axvline(x=best_result['threshold'], color='black', 
                   linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 设置坐标轴范围
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # 添加标签和标题
        ax2.set_xlabel('置信阈值', fontsize=12)
        ax2.set_ylabel('分数值', fontsize=12)
        ax2.set_title('SVM F0.3 分数', fontsize=14, fontweight='bold')
        
        # 添加图例
        ax2.legend(loc='lower left', fontsize=10)
        
        # 添加网格
        ax2.grid(True, alpha=0.3)
        
        # 调整子图间距
        plt.tight_layout()
        
        # 显示图形
        plt.show()
        
        # 返回图形对象以便保存
        return fig

def main():
    """主函数：寻找最优置信阈值并绘制简洁图表"""
    print("TF-IDF + SVM垃圾邮件检测系统 - 最优置信阈值搜索")
    print("=" * 50)
    
    # 1. 初始化分类器
    classifier = TFIDF_SVM_SpamClassifier(random_state=RANDOM_STATE)
    
    # 2. 加载和准备数据
    print("正在加载数据...")
    df = classifier.load_and_prepare_data('spam mail.csv')
    
    if df is None or len(df) == 0:
        print("无法加载数据，程序退出")
        return
    
    print(f"数据加载成功，共 {len(df)} 条样本")
    print(f"垃圾邮件比例: {df['label'].mean():.2%}")
    
    # 3. 划分数据集
    X_train, X_test, y_train, y_test = classifier.prepare_data(df, test_size=0.2)
    
    if X_train is None:
        print("数据划分失败，程序退出")
        return
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 4. 执行网格搜索训练模型
    print("正在训练模型...")
    grid_search = classifier.perform_grid_search(X_train, y_train, cv=5)
    
    print("模型训练完成")
    print(f"最佳参数: {classifier.best_params_}")
    print(f"交叉验证F1分数: {classifier.best_score_:.4f}")
    
    # 5. 寻找最优置信阈值
    print("\n正在搜索最优置信阈值以最大化F0.3分数...")
    best_result, all_results = classifier.find_optimal_threshold(
        X_test, y_test, 
        threshold_range=(0.1, 0.99), 
        num_points=100
    )
    
    # 6. 输出结果
    print("\n" + "=" * 50)
    print("最优阈值分析结果")
    print("=" * 50)
    print(f"最优置信阈值: {best_result['threshold']:.4f}")
    print(f"最大F0.3分数: {best_result['f03']:.4f}")
    print(f"对应精确率: {best_result['precision']:.4f}")
    print(f"对应召回率: {best_result['recall']:.4f}")
    print(f"对应F1分数: {best_result['f1']:.4f}")
    
    # 7. 绘制简洁图表
    print("\n正在生成可视化图表...")
    fig = classifier.plot_metrics_vs_threshold(all_results, best_result)
    
    # 8. 保存图表
    fig.savefig('svm_threshold_analysis.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'svm_threshold_analysis.png'")
    
    return classifier, best_result, all_results, fig

# 运行主程序
if __name__ == "__main__":
    try:
        classifier, best_result, all_results, fig = main()
    except FileNotFoundError:
        print("错误: 未找到 'spam mail.csv' 文件")
        print("请确保数据文件在当前目录下，或修改文件路径")
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")