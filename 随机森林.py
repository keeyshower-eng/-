import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TFIDF_RF_SpamClassifier:
    """基于TF-IDF和随机森林的垃圾邮件分类器"""
    
    def __init__(self, random_state=RANDOM_STATE, confidence_threshold=0.5):
        """
        初始化分类器
        
        参数:
        random_state: 随机种子
        confidence_threshold: 判定为垃圾邮件的置信度阈值
        """
        self.random_state = random_state
        self.confidence_threshold = confidence_threshold
        self.pipeline = None
        self.best_params_ = None
        self.best_score_ = None
        self.vectorizer = None
        self.classifier = None
        
    def load_and_prepare_data(self, filepath='spam mail.csv'):
        """
        加载和准备数据集
        
        参数:
        filepath: 数据文件路径
        
        返回:
        df: 处理后的DataFrame
        """
        print("正在加载数据集...")
        
        # 尝试不同的编码方式读取文件
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"成功使用 {encoding} 编码加载数据")
                break
            except UnicodeDecodeError:
                print(f"使用 {encoding} 编码失败，尝试下一个...")
                continue
            except FileNotFoundError:
                print(f"文件 {filepath} 未找到")
                return None
        
        if df is None:
            print("无法使用任何编码加载文件，请检查文件格式")
            return None
        
        print(f"数据集大小: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 检查并重命名列
        expected_columns = ['Category', 'Messages']
        for col in expected_columns:
            if col not in df.columns:
                # 尝试查找类似的列名
                for df_col in df.columns:
                    if col.lower() in df_col.lower():
                        print(f"将列 '{df_col}' 重命名为 '{col}'")
                        df = df.rename(columns={df_col: col})
                        break
        
        # 确保列名正确
        if 'Category' not in df.columns or 'Messages' not in df.columns:
            print("错误: 数据集必须包含 'Category' 和 'Messages' 列")
            print(f"实际列名: {df.columns.tolist()}")
            return None
        
        # 显示类别分布
        print("\n类别分布:")
        print(df['Category'].value_counts())
        
        # 将类别转换为数值标签
        category_mapping = {'spam': 1, 'ham': 0, 'Spam': 1, 'Ham': 0}
        df['label'] = df['Category'].map(category_mapping)
        
        # 检查是否有未映射的值
        if df['label'].isnull().any():
            print("警告: 发现未映射的类别值")
            unmapped = df[df['label'].isnull()]['Category'].unique()
            print(f"未映射的类别: {unmapped}")
            # 移除未映射的行
            df = df.dropna(subset=['label'])
        
        print(f"\n处理后数据集大小: {df.shape}")
        print(f"垃圾邮件数量: {(df['label'] == 1).sum()}")
        print(f"正常邮件数量: {(df['label'] == 0).sum()}")
        print(f"垃圾邮件比例: {df['label'].mean():.2%}")
        
        # 重命名列以便后续处理
        df = df.rename(columns={'Messages': 'text'})
        df['label'] = df['label'].astype(int)
        
        return df
    
    def preprocess_text(self, text):
        """
        文本预处理
        
        参数:
        text: 原始文本
        
        返回:
        预处理后的文本
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
        
        参数:
        df: 包含文本和标签的DataFrame
        test_size: 测试集比例
        
        返回:
        划分好的训练和测试数据
        """
        print(f"\n正在划分数据集 (训练集: {1-test_size:.0%}, 测试集: {test_size:.0%})...")
        print(f"随机种子: {self.random_state}")
        
        if df is None or len(df) == 0:
            print("错误: 数据为空")
            return None, None, None, None
        
        # 应用文本预处理
        print("正在预处理文本...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        if len(df) > 0:
            print(f"预处理后样本示例: {df['processed_text'].iloc[0][:100]}...")
        
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
        print(f"训练集中垃圾邮件比例: {y_train.mean():.2%}")
        print(f"测试集中垃圾邮件比例: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def create_pipeline(self):
        """
        创建TF-IDF + 随机森林的Pipeline
        
        返回:
        构建好的Pipeline
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('rf', RandomForestClassifier(random_state=self.random_state))
        ])
        
        return pipeline
    
    def perform_grid_search(self, X_train, y_train, cv=3):
        """
        执行网格搜索寻找最佳参数
        
        参数:
        X_train: 训练特征
        y_train: 训练标签
        cv: 交叉验证折数
        
        返回:
        网格搜索对象
        """
        print("\n正在执行网格搜索寻找最佳参数...")
        
        # 定义参数网格 - 针对随机森林优化
        param_grid = {
            'tfidf__max_features': [5000],
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__stop_words': ['english'],
            'tfidf__min_df': [2],
            'tfidf__max_df': [0.8],
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [None],
            'rf__min_samples_split': [5],
            'rf__min_samples_leaf': [1],
            'rf__class_weight': ['balanced']
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
            verbose=1,
            return_train_score=True
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        # 保存最佳参数和分数
        self.pipeline = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        
        # 获取最佳估计器的各个组件
        self.vectorizer = self.pipeline.named_steps['tfidf']
        self.classifier = self.pipeline.named_steps['rf']
        
        return grid_search
    
    def predict_with_threshold(self, X, threshold=None):
        """
        使用置信度阈值进行预测
        
        参数:
        X: 特征数据
        threshold: 分类阈值，如果为None则使用实例的confidence_threshold
        
        返回:
        预测标签、置信度和概率
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        # 获取预测概率
        proba = self.pipeline.predict_proba(X)
        
        # 获取垃圾邮件的概率
        spam_proba = proba[:, 1]
        
        # 只有当垃圾邮件概率大于阈值时才预测为垃圾邮件
        predictions = (spam_proba > threshold).astype(int)
        
        # 计算置信度（预测类别的概率）
        confidence = np.max(proba, axis=1)
        
        return predictions, confidence, spam_proba
    
    def find_best_threshold_for_fbeta(self, X_test, y_test, beta=0.3, threshold_range=None):
        """
        寻找使Fβ分数最大的最佳置信阈值
        
        参数:
        X_test: 测试特征
        y_test: 测试标签
        beta: Fβ分数中的β值
        threshold_range: 阈值搜索范围
        
        返回:
        最佳阈值和对应的评估指标
        """
        if threshold_range is None:
            threshold_range = np.arange(0.1, 1.0, 0.01)
        
        print(f"\n正在寻找使F{beta}分数最大的最佳置信阈值...")
        print(f"搜索范围: {threshold_range[0]:.2f} 到 {threshold_range[-1]:.2f}, 步长: 0.01")
        
        results = []
        
        for threshold in threshold_range:
            # 使用阈值进行预测
            y_pred, _, _ = self.predict_with_threshold(X_test, threshold)
            
            # 计算评估指标
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f_beta = fbeta_score(y_test, y_pred, beta=beta, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                f'f{beta}_score': f_beta,
                'accuracy': accuracy,
                'fpr': fpr
            })
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 找到Fβ分数最大的阈值
        best_idx = results_df[f'f{beta}_score'].idxmax()
        best_result = results_df.loc[best_idx]
        
        print(f"\n最佳阈值搜索结果:")
        print(f"最佳阈值: {best_result['threshold']:.4f}")
        print(f"对应的F{beta}分数: {best_result[f'f{beta}_score']:.4f}")
        print(f"精确率: {best_result['precision']:.4f}")
        print(f"召回率: {best_result['recall']:.4f}")
        print(f"准确率: {best_result['accuracy']:.4f}")
        print(f"误报率: {best_result['fpr']:.4f}")
        
        # 绘制评估指标随阈值变化的图表
        self.plot_threshold_analysis(results_df, beta)
        
        return best_result, results_df
    
    def plot_threshold_analysis(self, results_df, beta):
        """
        绘制精确率、召回率和Fβ分数随阈值变化的图表
        
        参数:
        results_df: 包含不同阈值下评估指标的DataFrame
        beta: Fβ分数中的β值
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 第一个子图：精确率和召回率
        ax1.plot(results_df['threshold'], results_df['precision'], 
                label='精确率 (Precision)', linewidth=2, color='blue')
        ax1.plot(results_df['threshold'], results_df['recall'], 
                label='召回率 (Recall)', linewidth=2, color='red')
        
        # 标记最佳阈值点
        best_threshold = results_df.loc[results_df[f'f{beta}_score'].idxmax(), 'threshold']
        best_precision = results_df.loc[results_df[f'f{beta}_score'].idxmax(), 'precision']
        best_recall = results_df.loc[results_df[f'f{beta}_score'].idxmax(), 'recall']
        
        ax1.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'最佳阈值 ({best_threshold:.3f})')
        ax1.plot(best_threshold, best_precision, 'bo', markersize=8)
        ax1.plot(best_threshold, best_recall, 'ro', markersize=8)
        
        ax1.set_xlabel('置信阈值', fontsize=12)
        ax1.set_ylabel('分数', fontsize=12)
        ax1.set_title(f'精确率和召回率随置信阈值变化 (F{beta}最优)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 第二个子图：Fβ分数
        ax2.plot(results_df['threshold'], results_df[f'f{beta}_score'], 
                label=f'F{beta}分数', linewidth=2, color='purple')
        
        # 标记最佳Fβ分数点
        best_fbeta = results_df.loc[results_df[f'f{beta}_score'].idxmax(), f'f{beta}_score']
        ax2.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'最佳阈值 ({best_threshold:.3f})')
        ax2.plot(best_threshold, best_fbeta, 'mo', markersize=8)
        
        ax2.set_xlabel('置信阈值', fontsize=12)
        ax2.set_ylabel(f'F{beta}分数', fontsize=12)
        ax2.set_title(f'F{beta}分数随置信阈值变化', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # 输出最佳阈值附近的详细信息
        print(f"\n最佳阈值 ({best_threshold:.3f}) 附近的性能:")
        threshold_window = 0.05
        nearby_results = results_df[
            (results_df['threshold'] >= best_threshold - threshold_window) & 
            (results_df['threshold'] <= best_threshold + threshold_window)
        ]
        
        print(f"阈值范围: {best_threshold - threshold_window:.3f} - {best_threshold + threshold_window:.3f}")
        print(f"F{beta}分数范围: {nearby_results[f'f{beta}_score'].min():.4f} - {nearby_results[f'f{beta}_score'].max():.4f}")
        print(f"精确率范围: {nearby_results['precision'].min():.4f} - {nearby_results['precision'].max():.4f}")
        print(f"召回率范围: {nearby_results['recall'].min():.4f} - {nearby_results['recall'].max():.4f}")

def main():
    """主函数：寻找使F0.3分数最大的最佳置信阈值"""
    print("="*60)
    print("TF-IDF + 随机森林垃圾邮件检测系统 - F0.3分数优化")
    print("="*60)
    
    # 1. 初始化分类器
    classifier = TFIDF_RF_SpamClassifier(random_state=RANDOM_STATE)
    
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
    
    # 4. 执行网格搜索寻找最佳参数
    grid_search = classifier.perform_grid_search(X_train, y_train, cv=3)
    
    # 5. 寻找使F0.3分数最大的最佳置信阈值
    beta = 0.3
    best_result, all_results = classifier.find_best_threshold_for_fbeta(
        X_test, y_test, beta=beta
    )
    
    # 6. 输出最终结果
    print("\n" + "="*60)
    print("最终结果总结")
    print("="*60)
    print(f"最佳置信阈值: {best_result['threshold']:.4f}")
    print(f"对应的F{beta}分数: {best_result[f'f{beta}_score']:.4f}")
    print(f"精确率: {best_result['precision']:.4f}")
    print(f"召回率: {best_result['recall']:.4f}")
    print(f"准确率: {best_result['accuracy']:.4f}")
    print(f"误报率: {best_result['fpr']:.4f}")
    
    return classifier, best_result, all_results

# 运行主程序
if __name__ == "__main__":
    try:
        classifier, best_result, all_results = main()
    except FileNotFoundError:
        print("错误: 未找到 'spam mail.csv' 文件")
        print("请确保数据集文件在当前目录中")
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()