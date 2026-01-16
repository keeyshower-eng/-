import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TFIDF_LR_SpamClassifier:
    """基于TF-IDF和逻辑回归的垃圾邮件分类器"""
    
    def __init__(self, random_state=RANDOM_STATE):
        """
        初始化分类器
        
        参数:
        random_state: 随机种子
        """
        self.random_state = random_state
        self.pipeline = None
        self.vectorizer = None
        self.classifier = None
        
    def load_and_prepare_data(self, filepath='spam mail.csv'):
        """
        加载和准备数据集
        """
        print("正在加载数据集...")
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"成功使用 {encoding} 编码加载数据")
                break
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                print(f"文件 {filepath} 未找到")
                return None
        
        if df is None:
            print("无法使用任何编码加载文件，请检查文件格式")
            return None
        
        # 检查列名
        expected_columns = ['Category', 'Messages']
        for col in expected_columns:
            if col not in df.columns:
                for df_col in df.columns:
                    if col.lower() in df_col.lower():
                        df = df.rename(columns={df_col: col})
                        break
        
        if 'Category' not in df.columns or 'Messages' not in df.columns:
            print("错误: 数据集必须包含 'Category' 和 'Messages' 列")
            return None
        
        # 将类别映射为数值
        category_mapping = {'spam': 1, 'ham': 0, 'Spam': 1, 'Ham': 0}
        df['Category'] = df['Category'].map(category_mapping)
        
        # 处理未映射的值
        if df['Category'].isnull().any():
            df['Category'] = df['Category'].fillna(-1)
        
        df = df[df['Category'] != -1]
        df = df.rename(columns={'Category': 'label', 'Messages': 'text'})
        df['label'] = df['label'].astype(int)
        
        print(f"数据集大小: {df.shape}")
        print(f"垃圾邮件数量: {(df['label'] == 1).sum()}")
        print(f"正常邮件数量: {(df['label'] == 0).sum()}")
        
        return df
    
    def preprocess_text(self, text):
        """
        文本预处理
        """
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        text = re.sub(r'\b\d{10,}\b', ' PHONE ', text)
        text = re.sub(r'[^\w\s\.\?\!,]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self, df, test_size=0.2):
        """
        准备训练和测试数据
        """
        print(f"\n正在划分数据集 (训练集: {1-test_size:.0%}, 测试集: {test_size:.0%})...")
        
        if df is None or len(df) == 0:
            print("错误: 数据为空")
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
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_and_train_model(self, X_train, y_train):
        """
        创建并训练模型
        """
        print("\n正在创建和训练模型...")
        
        # TF-IDF参数
        tfidf_params = {
            'max_features': 5000,
            'ngram_range': (1, 1),
            'stop_words': 'english',
            'min_df': 1,
            'max_df': 0.9
        }
        
        # 逻辑回归参数
        lr_params = {
            'C': 10.0,
            'penalty': 'l1',
            'solver': 'liblinear',
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'max_iter': 1000
        }
        
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # 创建并训练逻辑回归模型
        self.classifier = LogisticRegression(**lr_params)
        self.classifier.fit(X_train_tfidf, y_train)
        
        # 创建Pipeline用于后续预测
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('lr', self.classifier)
        ])
        
        print("模型训练完成!")
    
    def calculate_f_alpha(self, precision, recall, alpha=0.3):
        """
        计算Fα分数
        Fα = (1 + α²) * (precision * recall) / (α² * precision + recall)
        """
        if precision == 0 and recall == 0:
            return 0
        alpha_sq = alpha ** 2
        return (1 + alpha_sq) * (precision * recall) / (alpha_sq * precision + recall)
    
    def analyze_threshold_performance(self, X_test, y_test, alpha=0.3):
        """
        分析不同置信度阈值下的性能变化
        主要输出：精确率-召回率曲线和Fα变化曲线
        """
        print("\n" + "="*60)
        print("置信度阈值性能分析")
        print("="*60)
        
        if self.pipeline is None:
            print("请先训练模型")
            return None
        
        # 生成不同阈值
        thresholds = np.arange(0.1, 1.0, 0.05)
        
        # 存储结果
        results = []
        
        # 获取预测概率
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        print("正在计算不同阈值下的性能指标...")
        
        for threshold in thresholds:
            # 根据阈值进行预测
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # 计算评估指标
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f_alpha = self.calculate_f_alpha(precision, recall, alpha)
            
            # 计算误报率和漏报率
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f_alpha': f_alpha,
                'fpr': fpr,
                'fnr': fnr,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 1. 绘制精确率-召回率曲线
        self.plot_precision_recall_curve(results_df, alpha)
        
        # 2. 绘制Fα变化曲线
        self.plot_f_alpha_curve(results_df, alpha)
        
        # 找到最佳阈值（Fα最大化的阈值）
        best_idx = results_df['f_alpha'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_f_alpha = results_df.loc[best_idx, 'f_alpha']
        best_precision = results_df.loc[best_idx, 'precision']
        best_recall = results_df.loc[best_idx, 'recall']
        
        print(f"\n最佳阈值分析 (基于Fα最大化, α={alpha}):")
        print(f"最佳阈值: {best_threshold:.3f}")
        print(f"最大Fα分数: {best_f_alpha:.4f}")
        print(f"对应精确率: {best_precision:.4f}")
        print(f"对应召回率: {best_recall:.4f}")
        
        # 显示阈值选择建议
        print(f"\n阈值选择建议:")
        high_precision_thresh = results_df[results_df['precision'] >= 0.95]['threshold'].min()
        balanced_thresh = results_df[results_df['f_alpha'] >= best_f_alpha * 0.95]['threshold'].min()
        
        if not pd.isna(high_precision_thresh):
            high_precision_row = results_df[results_df['threshold'] == high_precision_thresh].iloc[0]
            print(f"- 高精确率阈值 ({high_precision_thresh:.3f}): 精确率={high_precision_row['precision']:.4f}, 召回率={high_precision_row['recall']:.4f}")
        
        print(f"- 平衡阈值 ({balanced_thresh:.3f}): Fα={results_df[results_df['threshold'] == balanced_thresh]['f_alpha'].iloc[0]:.4f}")
        
        return results_df
    
    def plot_precision_recall_curve(self, results_df, alpha=0.3):
        """
        绘制精确率-召回率曲线
        """
        plt.figure(figsize=(10, 8))
        
        # 精确率-召回率曲线
        plt.plot(results_df['recall'], results_df['precision'], 
                marker='o', linewidth=2, markersize=6, label='P-R曲线')
        
        # 标记几个关键阈值点
        key_thresholds = [0.3, 0.5, 0.7, 0.9]
        for thresh in key_thresholds:
            closest_idx = (results_df['threshold'] - thresh).abs().idxmin()
            row = results_df.loc[closest_idx]
            plt.plot(row['recall'], row['precision'], 'ro', markersize=8)
            plt.annotate(f'thresh={thresh}', 
                        (row['recall'], row['precision']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title(f'精确率-召回率曲线 (α={alpha})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        
        # 绘制阈值-精确率/召回率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['threshold'], results_df['precision'], 
                marker='o', linewidth=2, label='精确率')
        plt.plot(results_df['threshold'], results_df['recall'], 
                marker='s', linewidth=2, label='召回率')
        
        plt.xlabel('置信度阈值', fontsize=12)
        plt.ylabel('分数', fontsize=12)
        plt.title('精确率和召回率随阈值变化曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_f_alpha_curve(self, results_df, alpha=0.3):
        """
        绘制Fα变化曲线
        """
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 子图1: Fα曲线
        ax1.plot(results_df['threshold'], results_df['f_alpha'], 
                marker='o', linewidth=3, markersize=6, color='green', label=f'Fα (α={alpha})')
        
        # 标记最大值
        best_idx = results_df['f_alpha'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_f_alpha = results_df.loc[best_idx, 'f_alpha']
        
        ax1.plot(best_threshold, best_f_alpha, 'ro', markersize=10, 
                label=f'最大值: Fα={best_f_alpha:.3f} (阈值={best_threshold:.3f})')
        
        ax1.set_xlabel('置信度阈值', fontsize=12)
        ax1.set_ylabel(f'Fα分数 (α={alpha})', fontsize=12)
        ax1.set_title(f'Fα分数随置信度阈值变化 (α={alpha})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 子图2: 误报率和漏报率
        ax2.plot(results_df['threshold'], results_df['fpr'], 
                marker='^', linewidth=2, label='误报率 (FPR)')
        ax2.plot(results_df['threshold'], results_df['fnr'], 
                marker='v', linewidth=2, label='漏报率 (FNR)')
        
        ax2.set_xlabel('置信度阈值', fontsize=12)
        ax2.set_ylabel('错误率', fontsize=12)
        ax2.set_title('误报率和漏报率随阈值变化', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 显示Fα的详细分析
        print(f"\nFα分数分析 (α={alpha}):")
        print("=" * 50)
        
        # 显示不同阈值区间的Fα表现
        threshold_ranges = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
        
        for low, high in threshold_ranges:
            mask = (results_df['threshold'] >= low) & (results_df['threshold'] < high)
            if mask.any():
                subset = results_df[mask]
                max_f_alpha = subset['f_alpha'].max()
                best_in_range = subset[subset['f_alpha'] == max_f_alpha].iloc[0]
                
                print(f"阈值范围 [{low:.1f}-{high:.1f}):")
                print(f"  最大Fα: {max_f_alpha:.4f} (阈值={best_in_range['threshold']:.3f})")
                print(f"  精确率: {best_in_range['precision']:.4f}")
                print(f"  召回率: {best_in_range['recall']:.4f}")
                print()

def main():
    """主函数：运行置信度阈值分析"""
    print("="*60)
    print("垃圾邮件分类器 - 置信度阈值性能分析")
    print("重点分析: 精确率-召回率曲线和Fα(α=0.3)变化")
    print("="*60)
    
    # 1. 初始化分类器
    classifier = TFIDF_LR_SpamClassifier(random_state=RANDOM_STATE)
    
    # 2. 加载和准备数据
    df = classifier.load_and_prepare_data('spam mail.csv')
    
    if df is None or len(df) == 0:
        print("无法加载数据，程序退出")
        return
    
    # 3. 划分数据集
    X_train, X_test, y_train, y_test = classifier.prepare_data(df, test_size=0.2)
    
    if X_train is None:
        print("数据划分失败，程序退出")
        return
    
    # 4. 创建并训练模型
    classifier.create_and_train_model(X_train, y_train)
    
    # 5. 进行置信度阈值分析 (α=0.3)
    alpha = 0.3
    print(f"\n开始置信度阈值分析 (α={alpha})...")
    results = classifier.analyze_threshold_performance(X_test, y_test, alpha=alpha)
    
    if results is not None:
        # 显示性能总结
        print("\n" + "="*60)
        print("性能总结")
        print("="*60)
        
        best_idx = results['f_alpha'].idxmax()
        best_row = results.loc[best_idx]
        
        print(f"最佳阈值: {best_row['threshold']:.3f}")
        print(f"最大Fα分数: {best_row['f_alpha']:.4f}")
        print(f"对应精确率: {best_row['precision']:.4f}")
        print(f"对应召回率: {best_row['recall']:.4f}")
        print(f"误报率: {best_row['fpr']:.4f}")
        print(f"漏报率: {best_row['fnr']:.4f}")
        
        # 显示不同应用场景的建议
        print(f"\n应用场景建议:")
        print("1. 高精确率需求 (减少误报): 选择阈值 > 0.7")
        print("2. 平衡需求: 选择阈值 0.4-0.6")
        print("3. 高召回率需求 (减少漏报): 选择阈值 < 0.3")
    
    return classifier, results

# 运行主程序
if __name__ == "__main__":
    try:
        classifier, results = main()
    except FileNotFoundError:
        print("错误: 未找到 'spam mail.csv' 文件")
        print("请确保数据集文件在当前目录中")
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()