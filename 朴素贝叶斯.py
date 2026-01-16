import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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

class TFIDF_NB_SpamClassifier:
    """基于TF-IDF和朴素贝叶斯的垃圾邮件分类器"""
    
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
        # 读取数据，假设数据集有两列：Category和Messages
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, encoding='latin-1')
            except:
                print("错误: 无法读取文件，请检查文件编码")
                return None
        
        # 检查并重命名列
        df = df.rename(columns={
            'Category': 'category',
            'Messages': 'message',
            'Masseges': 'message',  # 处理可能的拼写错误
            'category': 'category',
            'message': 'message'
        })
        
        # 确保有正确的列
        if 'category' not in df.columns or 'message' not in df.columns:
            print("错误: 数据集必须包含 'Category' 和 'Messages' 列")
            return None
        
        # 将类别转换为数值标签
        df['label'] = df['category'].map({'ham': 0, 'spam': 1, 'Ham': 0, 'Spam': 1})
        
        # 检查是否有未映射的值
        if df['label'].isnull().any():
            # 移除未映射的行
            df = df.dropna(subset=['label'])
        
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
        # 应用文本预处理
        df['processed_text'] = df['message'].apply(self.preprocess_text)
        
        # 划分数据集，设置随机种子确保可重复性
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
        创建TF-IDF + 朴素贝叶斯的Pipeline
        
        返回:
        构建好的Pipeline
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('nb', MultinomialNB())
        ])
        
        return pipeline
    
    def perform_grid_search(self, X_train, y_train, cv=5):
        """
        执行网格搜索寻找最佳参数
        
        参数:
        X_train: 训练特征
        y_train: 训练标签
        cv: 交叉验证折数
        
        返回:
        网格搜索对象
        """
        # 定义参数网格
        param_grid = {
            'tfidf__max_features': [5000],
            'tfidf__ngram_range': [(1, 3)],
            'tfidf__stop_words': ['english'],
            'tfidf__min_df': [1],
            'tfidf__max_df': [0.9],
            'nb__alpha': [1.0],
            'nb__fit_prior': [True]
        }
        
        # 创建Pipeline
        pipeline = self.create_pipeline()
        
        # 创建GridSearchCV对象
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring='precision',
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
        
        # 获取最佳估计器的各个组件
        self.vectorizer = self.pipeline.named_steps['tfidf']
        self.classifier = self.pipeline.named_steps['nb']
        
        return grid_search
    
    def predict_with_threshold(self, X, threshold=None):
        """
        使用置信度阈值进行预测
        
        参数:
        X: 特征数据
        threshold: 分类阈值，如果为None则使用实例的confidence_threshold
        
        返回:
        预测标签和置信度
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
        寻找使F-beta分数最大的最佳置信阈值
        
        参数:
        X_test: 测试特征
        y_test: 测试标签
        beta: F-beta分数中的beta值
        threshold_range: 阈值搜索范围
        
        返回:
        最佳阈值和对应的指标
        """
        if threshold_range is None:
            threshold_range = np.linspace(0.1, 0.99, 50)
        
        results = []
        
        for threshold in threshold_range:
            # 使用阈值进行预测
            y_pred, _, _ = self.predict_with_threshold(X_test, threshold)
            
            # 计算评估指标
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f_beta = fbeta_score(y_test, y_pred, beta=beta, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                f'f_{beta}_score': f_beta
            })
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 找到F-beta分数最大的阈值
        best_idx = results_df[f'f_{beta}_score'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_f_beta = results_df.loc[best_idx, f'f_{beta}_score']
        best_precision = results_df.loc[best_idx, 'precision']
        best_recall = results_df.loc[best_idx, 'recall']
        
        # 输出最佳结果
        print("="*60)
        print(f"寻找使F{beta}分数最大的最佳置信阈值")
        print("="*60)
        print(f"最佳置信阈值: {best_threshold:.4f}")
        print(f"对应的F{beta}分数: {best_f_beta:.4f}")
        print(f"对应的精确率: {best_precision:.4f}")
        print(f"对应的召回率: {best_recall:.4f}")
        
        # 设置最佳阈值
        self.confidence_threshold = best_threshold
        
        return best_threshold, best_f_beta, best_precision, best_recall, results_df
    
    def plot_metrics_vs_threshold(self, results_df, beta=0.3):
        """
        绘制精确率、召回率和F-beta分数随阈值变化的折线图
        
        参数:
        results_df: 包含不同阈值下指标结果的DataFrame
        beta: F-beta分数中的beta值
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 第一个图：精确率和召回率
        ax1.plot(results_df['threshold'], results_df['precision'], 
                label='精确率 (Precision)', linewidth=2, color='blue')
        ax1.plot(results_df['threshold'], results_df['recall'], 
                label='召回率 (Recall)', linewidth=2, color='red')
        ax1.set_xlabel('置信阈值')
        ax1.set_ylabel('分数')
        ax1.set_title('精确率和召回率 vs. 置信阈值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标记最佳阈值点
        best_threshold = self.confidence_threshold
        best_idx = results_df[results_df['threshold'] == best_threshold].index[0]
        best_precision = results_df.loc[best_idx, 'precision']
        best_recall = results_df.loc[best_idx, 'recall']
        
        ax1.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7)
        ax1.plot(best_threshold, best_precision, 'bo', markersize=8)
        ax1.plot(best_threshold, best_recall, 'ro', markersize=8)
        ax1.text(best_threshold, 0.5, f'最佳阈值\n{best_threshold:.3f}', 
                horizontalalignment='center', verticalalignment='bottom')
        
        # 第二个图：F-beta分数
        ax2.plot(results_df['threshold'], results_df[f'f_{beta}_score'], 
                label=f'F{beta}分数', linewidth=2, color='purple')
        ax2.set_xlabel('置信阈值')
        ax2.set_ylabel(f'F{beta}分数')
        ax2.set_title(f'F{beta}分数 vs. 置信阈值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 标记最佳阈值点
        best_f_beta = results_df.loc[best_idx, f'f_{beta}_score']
        ax2.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7)
        ax2.plot(best_threshold, best_f_beta, 'mo', markersize=8)
        ax2.text(best_threshold, best_f_beta/2, f'最佳阈值\n{best_threshold:.3f}', 
                horizontalalignment='center', verticalalignment='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # 输出最佳阈值附近的详细指标
        print(f"\n最佳阈值 ({best_threshold:.4f}) 附近的详细指标:")
        print("-" * 50)
        
        # 找到最佳阈值附近的几个点
        threshold_diff = np.abs(results_df['threshold'] - best_threshold)
        nearby_indices = threshold_diff.nsmallest(5).index
        
        for idx in nearby_indices:
            threshold = results_df.loc[idx, 'threshold']
            precision = results_df.loc[idx, 'precision']
            recall = results_df.loc[idx, 'recall']
            f_beta = results_df.loc[idx, f'f_{beta}_score']
            
            marker = "★" if threshold == best_threshold else ""
            print(f"阈值: {threshold:.4f} {marker} - 精确率: {precision:.4f}, "
                  f"召回率: {recall:.4f}, F{beta}: {f_beta:.4f}")

def main():
    """主函数：寻找使F0.3分数最大的最佳置信阈值"""
    print("="*60)
    print("TF-IDF + 朴素贝叶斯垃圾邮件检测系统")
    print("目标：寻找使F0.3分数最大的最佳置信阈值")
    print("="*60)
    
    # 1. 初始化分类器
    classifier = TFIDF_NB_SpamClassifier(random_state=RANDOM_STATE)
    
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
    
    # 4. 执行网格搜索训练模型
    grid_search = classifier.perform_grid_search(X_train, y_train, cv=5)
    
    # 5. 寻找使F0.3分数最大的最佳置信阈值
    beta = 0.3
    best_threshold, best_f_beta, best_precision, best_recall, results_df = \
        classifier.find_best_threshold_for_fbeta(X_test, y_test, beta=beta)
    
    # 6. 绘制指标随阈值变化的图表
    classifier.plot_metrics_vs_threshold(results_df, beta=beta)
    
    # 7. 在最佳阈值下评估模型性能
    print("\n" + "="*60)
    print(f"在最佳阈值 ({best_threshold:.4f}) 下的模型性能")
    print("="*60)
    
    # 使用最佳阈值进行预测
    y_pred, _, _ = classifier.predict_with_threshold(X_test, best_threshold)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f_beta = fbeta_score(y_test, y_pred, beta=beta, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F{beta}分数:       {f_beta:.4f}")
    print(f"F1分数:           {f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n误报分析:")
    print(f"误报数 (FP): {fp}")
    print(f"误报率 (FPR): {fpr:.4f}")
    print(f"漏报数 (FN): {fn}")
    print(f"漏报率 (FNR): {fnr:.4f}")
    
    # 8. 与默认阈值(0.5)的比较
    print("\n" + "="*60)
    print("与默认阈值(0.5)的性能比较")
    print("="*60)
    
    # 使用默认阈值进行预测
    y_pred_default, _, _ = classifier.predict_with_threshold(X_test, 0.5)
    
    precision_default = precision_score(y_test, y_pred_default, zero_division=0)
    recall_default = recall_score(y_test, y_pred_default, zero_division=0)
    f_beta_default = fbeta_score(y_test, y_pred_default, beta=beta, zero_division=0)
    
    comparison = pd.DataFrame({
        '指标': ['精确率', '召回率', f'F{beta}分数'],
        '阈值=0.5': [precision_default, recall_default, f_beta_default],
        f'阈值={best_threshold:.4f}': [precision, recall, f_beta],
        '变化': [precision - precision_default, recall - recall_default, f_beta - f_beta_default]
    })
    
    print(comparison.to_string(index=False, float_format='%.4f'))
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)
    
    return classifier, best_threshold, best_f_beta, best_precision, best_recall

# 运行主程序
if __name__ == "__main__":
    try:
        classifier, best_threshold, best_f_beta, best_precision, best_recall = main()
    except FileNotFoundError:
        print("错误: 未找到 'spam mail.csv' 文件")
        print("请确保数据集文件在当前目录中")
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()