import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 导入深度学习相关库
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 设置随机种子确保可重复性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TextCNN_SpamClassifier:
    """基于TextCNN的垃圾邮件分类器"""
    
    def __init__(self, random_state=RANDOM_STATE, confidence_threshold=0.5,
                 max_features=5000, max_len=100, embedding_dim=100,
                 filters=128, kernel_sizes=[3, 4, 5], dropout_rate=0.5):
        """
        初始化分类器
        
        参数:
        random_state: 随机种子
        confidence_threshold: 判定为垃圾邮件的置信度阈值
        max_features: 最大词汇量
        max_len: 文本最大长度
        embedding_dim: 词向量维度
        filters: 卷积核数量
        kernel_sizes: 卷积核大小列表
        dropout_rate: Dropout比率
        """
        self.random_state = random_state
        self.confidence_threshold = confidence_threshold
        self.max_features = max_features
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        
        self.tokenizer = None
        self.model = None
        self.history = None
        self.label_encoder = LabelEncoder()
        
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
        
        # 去除特殊字符，保留字母、数字和基本标点
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!,]', ' ', text)
        
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
        
        # 检查预处理后的数据
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
    
    def create_textcnn_model(self, vocab_size):
        """
        创建TextCNN模型
        
        参数:
        vocab_size: 词汇表大小
        
        返回:
        构建好的TextCNN模型
        """
        print("\n正在创建TextCNN模型...")
        print(f"模型参数:")
        print(f"  词汇表大小: {vocab_size}")
        print(f"  最大序列长度: {self.max_len}")
        print(f"  词向量维度: {self.embedding_dim}")
        print(f"  卷积核数量: {self.filters}")
        print(f"  卷积核大小: {self.kernel_sizes}")
        print(f"  Dropout比率: {self.dropout_rate}")
        
        # 输入层
        inputs = Input(shape=(self.max_len,), dtype='int32')
        
        # 嵌入层
        embedding = Embedding(
            input_dim=vocab_size + 1,  # +1 for padding
            output_dim=self.embedding_dim,
            input_length=self.max_len,
            mask_zero=False
        )(inputs)
        
        # 多尺寸卷积层
        conv_blocks = []
        for kernel_size in self.kernel_sizes:
            conv = Conv1D(
                filters=self.filters,
                kernel_size=kernel_size,
                padding='valid',
                activation='relu',
                strides=1
            )(embedding)
            conv = GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
        
        # 合并不同尺寸的卷积结果
        if len(conv_blocks) > 1:
            concat = Concatenate()(conv_blocks)
        else:
            concat = conv_blocks[0]
        
        # 全连接层
        dropout = Dropout(self.dropout_rate)(concat)
        outputs = Dense(1, activation='sigmoid')(dropout)  # 二分类，使用sigmoid激活函数
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("TextCNN模型创建完成!")
        model.summary()
        
        return model
    
    def prepare_sequences(self, texts, fit_tokenizer=False):
        """
        将文本转换为序列
        
        参数:
        texts: 文本列表
        fit_tokenizer: 是否拟合tokenizer
        
        返回:
        填充后的序列
        """
        if fit_tokenizer or self.tokenizer is None:
            print("正在创建和拟合tokenizer...")
            self.tokenizer = Tokenizer(
                num_words=self.max_features,
                oov_token='<OOV>',
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
            self.tokenizer.fit_on_texts(texts)
            print(f"词汇表大小: {len(self.tokenizer.word_index)}")
        
        # 将文本转换为序列
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # 对序列进行填充
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_len, 
            padding='post', 
            truncating='post'
        )
        
        return padded_sequences
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   epochs=20, batch_size=32, validation_split=0.1):
        """
        训练TextCNN模型
        
        参数:
        X_train: 训练文本
        y_train: 训练标签
        X_val: 验证文本（可选）
        y_val: 验证标签（可选）
        epochs: 训练轮数
        batch_size: 批大小
        validation_split: 验证集比例（如果没有提供验证集）
        
        返回:
        训练历史
        """
        print("\n正在准备训练数据...")
        
        # 准备训练序列
        X_train_seq = self.prepare_sequences(X_train, fit_tokenizer=True)
        
        # 准备验证数据
        if X_val is not None and y_val is not None:
            X_val_seq = self.prepare_sequences(X_val)
            validation_data = (X_val_seq, y_val)
            print(f"使用独立验证集，大小: {len(X_val)}")
        else:
            validation_data = None
            validation_split = validation_split
            print(f"使用训练集分割验证集，比例: {validation_split:.0%}")
        
        print(f"训练序列形状: {X_train_seq.shape}")
        print(f"训练标签形状: {y_train.shape}")
        
        # 创建模型
        vocab_size = min(self.max_features, len(self.tokenizer.word_index))
        self.model = self.create_textcnn_model(vocab_size)
        
        # 设置回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data or validation_split else 'loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data or validation_split else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("\n开始训练TextCNN模型...")
        print(f"训练轮数: {epochs}")
        print(f"批大小: {batch_size}")
        
        # 训练模型
        self.history = self.model.fit(
            X_train_seq, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            validation_split=validation_split if not validation_data else 0.0,
            callbacks=callbacks,
            verbose=1
        )
        
        print("TextCNN模型训练完成!")
        
        return self.history
    
    def predict_with_threshold(self, X, threshold=None):
        """
        使用置信度阈值进行预测
        
        参数:
        X: 特征数据（文本）
        threshold: 分类阈值，如果为None则使用实例的confidence_threshold
        
        返回:
        预测标签、置信度和概率
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未训练")
        
        # 准备序列
        X_seq = self.prepare_sequences(X)
        
        # 获取预测概率
        proba = self.model.predict(X_seq, verbose=0)
        
        # 获取垃圾邮件的概率
        spam_proba = proba.flatten()
        
        # 只有当垃圾邮件概率大于阈值时才预测为垃圾邮件
        predictions = (spam_proba > threshold).astype(int)
        
        # 计算置信度（预测类别的概率）
        confidence = np.where(predictions == 1, spam_proba, 1 - spam_proba)
        
        return predictions, confidence, spam_proba
    
    def evaluate_model(self, X_test, y_test, threshold=None):
        """
        评估模型性能
        
        参数:
        X_test: 测试特征
        y_test: 测试标签
        threshold: 分类阈值，如果为None则使用实例的confidence_threshold
        
        返回:
        评估指标字典
        """
        if threshold is None:
            threshold = self.confidence_threshold
            
        print("\n" + "="*60)
        print(f"TextCNN模型性能评估")
        print(f"置信度阈值: {threshold}")
        print("="*60)
        
        # 使用阈值进行预测
        y_pred, confidence, y_pred_proba = self.predict_with_threshold(X_test, threshold)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f03 = fbeta_score(y_test, y_pred, beta=0.3, zero_division=0)  # F0.3分数
        
        print(f"准确率 (Accuracy):  {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f} (垃圾邮件预测的准确率)")
        print(f"召回率 (Recall):    {recall:.4f} (垃圾邮件被正确识别的比例)")
        print(f"F1分数:            {f1:.4f}")
        print(f"F0.3分数:          {f03:.4f} (更重视精确率的F分数)")
        print(f"平均置信度:        {confidence.mean():.4f}")
        
        # 详细分类报告
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['正常邮件 (Ham)', '垃圾邮件 (Spam)'],
                                   zero_division=0))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)
        
        # ROC曲线
        self.plot_roc_curve(y_test, y_pred_proba, threshold)
        
        # 计算误报率和漏报率
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\n误报分析:")
        print(f"正常邮件总数: {tn + fp}")
        print(f"误报数 (FP): {fp}")
        print(f"误报率 (FPR): {fpr:.4f} (正常邮件被误判为垃圾邮件的比例)")
        print(f"漏报数 (FN): {fn}")
        print(f"漏报率 (FNR): {fnr:.4f} (垃圾邮件被漏判为正常邮件的比例)")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f03': f03,  # 添加F0.3分数
            'confusion_matrix': cm,
            'y_pred_proba': y_pred_proba,
            'threshold': threshold,
            'fpr': fpr,
            'fnr': fnr
        }
    
    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['预测正常', '预测垃圾'],
                    yticklabels=['实际正常', '实际垃圾'])
        plt.title('TextCNN模型 - 混淆矩阵', fontsize=14)
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # 混淆矩阵详细信息
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n混淆矩阵详情:")
        print(f"真阴性 (TN): {tn} - 正常邮件被正确识别")
        print(f"假阳性 (FP): {fp} - 正常邮件被误判为垃圾邮件")
        print(f"假阴性 (FN): {fn} - 垃圾邮件被误判为正常邮件")
        print(f"真阳性 (TP): {tp} - 垃圾邮件被正确识别")
        
        # 计算特异性
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"特异性 (Specificity): {specificity:.4f} (正常邮件被正确识别的比例)")
    
    def plot_roc_curve(self, y_true, y_scores, threshold):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
        
        # 标记当前阈值对应的点
        fpr_threshold = fpr[np.argmin(np.abs(thresholds - threshold))]
        tpr_threshold = tpr[np.argmin(np.abs(thresholds - threshold))]
        plt.plot(fpr_threshold, tpr_threshold, 'ro', markersize=8, 
                 label=f'阈值 {threshold:.2f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        plt.title('TextCNN模型 - ROC曲线', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"ROC曲线AUC面积: {roc_auc:.4f}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可显示")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制损失曲线
        ax1.plot(self.history.history['loss'], label='训练损失')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='验证损失')
        ax1.set_title('模型损失')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制准确率曲线
        ax2.plot(self.history.history['accuracy'], label='训练准确率')
        if 'val_accuracy' in self.history.history:
            ax2.plot(self.history.history['val_accuracy'], label='验证准确率')
        ax2.set_title('模型准确率')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_f03_threshold_analysis(self, X_test, y_test, threshold_range=None):
        """
        绘制F0.3分数随阈值变化的分析图
        
        参数:
        X_test: 测试特征
        y_test: 测试标签
        threshold_range: 阈值范围
        """
        if threshold_range is None:
            threshold_range = np.arange(0.1, 1.0, 0.01)
        
        print(f"\n正在分析F0.3分数随阈值变化...")
        print(f"阈值搜索范围: {threshold_range[0]:.2f} 到 {threshold_range[-1]:.2f}, 步长: 0.01")
        
        # 存储不同阈值下的指标
        results = []
        
        for threshold in threshold_range:
            # 使用阈值进行预测
            y_pred, _, _ = self.predict_with_threshold(X_test, threshold)
            
            # 计算评估指标
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f03 = fbeta_score(y_test, y_pred, beta=0.3, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f03_score': f03,
                'accuracy': accuracy
            })
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 找到F0.3分数最大的阈值
        best_idx = results_df['f03_score'].idxmax()
        best_result = results_df.loc[best_idx]
        
        # 绘制图表
        self._plot_f03_analysis_charts(results_df, best_result)
        
        return best_result, results_df
    
    def _plot_f03_analysis_charts(self, results_df, best_result):
        """
        绘制F0.3分析图表
        
        参数:
        results_df: 包含结果的DataFrame
        best_result: 最佳结果
        """
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 第一个子图：精确率和召回率
        ax1.plot(results_df['threshold'], results_df['precision'], 
                label='精确率 (Precision)', linewidth=2, color='blue')
        ax1.plot(results_df['threshold'], results_df['recall'], 
                label='召回率 (Recall)', linewidth=2, color='red')
        
        # 标记最佳阈值点
        best_threshold = best_result['threshold']
        best_precision = best_result['precision']
        best_recall = best_result['recall']
        
        ax1.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'最佳阈值 ({best_threshold:.3f})')
        ax1.plot(best_threshold, best_precision, 'bo', markersize=8)
        ax1.plot(best_threshold, best_recall, 'ro', markersize=8)
        
        ax1.set_xlabel('置信阈值', fontsize=12)
        ax1.set_ylabel('分数', fontsize=12)
        ax1.set_title('精确率和召回率随置信阈值变化 (F0.3最优)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 第二个子图：F0.3分数
        ax2.plot(results_df['threshold'], results_df['f03_score'], 
                label='F0.3分数', linewidth=2, color='purple')
        
        # 标记最佳F0.3分数点
        best_f03 = best_result['f03_score']
        ax2.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'最佳阈值 ({best_threshold:.3f})')
        ax2.plot(best_threshold, best_f03, 'mo', markersize=8)
        
        ax2.set_xlabel('置信阈值', fontsize=12)
        ax2.set_ylabel('F0.3分数', fontsize=12)
        ax2.set_title('F0.3分数随置信阈值变化', fontsize=14)
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
        print(f"F0.3分数范围: {nearby_results['f03_score'].min():.4f} - {nearby_results['f03_score'].max():.4f}")
        print(f"精确率范围: {nearby_results['precision'].min():.4f} - {nearby_results['precision'].max():.4f}")
        print(f"召回率范围: {nearby_results['recall'].min():.4f} - {nearby_results['recall'].max():.4f}")
        
        # 显示最佳结果
        print(f"\n最佳F0.3分数结果:")
        print(f"最佳阈值: {best_threshold:.4f}")
        print(f"最佳F0.3分数: {best_f03:.4f}")
        print(f"精确率: {best_precision:.4f}")
        print(f"召回率: {best_recall:.4f}")
        print(f"准确率: {best_result['accuracy']:.4f}")
    
    def plot_confidence_distribution(self, X_test, y_test, threshold=None):
        """
        绘制置信度分布图
        
        参数:
        X_test: 测试特征
        y_test: 测试标签
        threshold: 分类阈值
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        # 获取预测概率
        _, _, y_pred_proba = self.predict_with_threshold(X_test, threshold)
        
        # 分离正常邮件和垃圾邮件的概率
        ham_proba = y_pred_proba[y_test == 0]
        spam_proba_true = y_pred_proba[y_test == 1]
        
        plt.figure(figsize=(12, 5))
        
        # 子图1: 概率分布直方图
        plt.subplot(1, 2, 1)
        plt.hist(ham_proba, bins=50, alpha=0.7, label='正常邮件', color='blue', density=True)
        plt.hist(spam_proba_true, bins=50, alpha=0.7, label='垃圾邮件', color='red', density=True)
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'阈值={threshold}')
        plt.xlabel('垃圾邮件概率', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        plt.title('TextCNN模型 - 垃圾邮件概率分布', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 累积分布函数
        plt.subplot(1, 2, 2)
        sorted_ham = np.sort(ham_proba)
        sorted_spam = np.sort(spam_proba_true)
        y_vals_ham = np.arange(len(sorted_ham)) / float(len(sorted_ham))
        y_vals_spam = np.arange(len(sorted_spam)) / float(len(sorted_spam))
        
        plt.plot(sorted_ham, y_vals_ham, label='正常邮件', color='blue', linewidth=2)
        plt.plot(sorted_spam, y_vals_spam, label='垃圾邮件', color='red', linewidth=2)
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'阈值={threshold}')
        plt.xlabel('垃圾邮件概率', fontsize=12)
        plt.ylabel('累积概率', fontsize=12)
        plt.title('TextCNN模型 - 累积分布函数', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 计算阈值处的分类情况
        ham_above_threshold = np.sum(ham_proba > threshold)
        spam_below_threshold = np.sum(spam_proba_true < threshold)
        
        print(f"正常邮件中，概率大于阈值 {threshold} 的比例: {ham_above_threshold/len(ham_proba):.2%}")
        print(f"垃圾邮件中，概率小于阈值 {threshold} 的比例: {spam_below_threshold/len(spam_proba_true):.2%}")
    
    def find_best_f03_threshold(self, X_test, y_test, threshold_start=0.5, threshold_step=0.01, max_threshold=0.99):
        """
        寻找使F0.3分数最大的阈值
        
        参数:
        X_test: 测试特征
        y_test: 测试标签
        threshold_start: 起始阈值
        threshold_step: 阈值步长
        max_threshold: 最大阈值
        
        返回:
        最佳阈值和对应的评估指标
        """
        print("\n" + "="*60)
        print("寻找使F0.3分数最大的最佳阈值")
        print("="*60)
        
        # 使用更精细的阈值搜索
        threshold_range = np.arange(threshold_start, max_threshold + threshold_step, threshold_step)
        best_result, all_results = self.plot_f03_threshold_analysis(X_test, y_test, threshold_range)
        
        self.confidence_threshold = best_result['threshold']
        return best_result['threshold'], best_result
    
    def predict_example(self, text, threshold=None):
        """
        对新邮件进行预测
        
        参数:
        text: 邮件文本
        threshold: 分类阈值
        
        返回:
        预测结果
        """
        if self.model is None or self.tokenizer is None:
            print("请先训练模型")
            return None
        
        if threshold is None:
            threshold = self.confidence_threshold
        
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 将文本转换为序列
        text_seq = self.prepare_sequences([processed_text])
        
        # 进行预测
        proba = self.model.predict(text_seq, verbose=0)[0][0]
        spam_probability = proba
        ham_probability = 1 - proba
        
        # 使用阈值进行分类
        prediction = 1 if spam_probability > threshold else 0
        
        # 计算置信度
        confidence = spam_probability if prediction == 1 else ham_probability
        
        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': '垃圾邮件 (Spam)' if prediction == 1 else '正常邮件 (Ham)',
            'spam_probability': spam_probability,
            'ham_probability': ham_probability,
            'threshold_used': threshold,
            'confidence': confidence,
            'is_spam': prediction == 1
        }
        
        print(f"\n邮件内容: {result['text']}")
        print(f"预测结果: {result['prediction']}")
        print(f"垃圾邮件概率: {result['spam_probability']:.4f}")
        print(f"正常邮件概率: {result['ham_probability']:.4f}")
        print(f"使用阈值: {result['threshold_used']:.2f}")
        print(f"预测置信度: {result['confidence']:.4f}")
        
        # 判断是否达到置信度阈值
        if prediction == 1:
            if spam_probability >= threshold + 0.1:
                print(f"✓ 高置信度垃圾邮件 (置信度 > 阈值 + 0.1)")
            elif spam_probability >= threshold:
                print(f"✓ 中等置信度垃圾邮件 (置信度接近阈值)")
            else:
                print(f"✗✗ 未达到阈值，判定为正常邮件")
        else:
            if spam_probability < threshold - 0.1:
                print(f"✓ 高置信度正常邮件 (垃圾邮件概率 < 阈值 - 0.1)")
            else:
                print(f"✓ 中等置信度正常邮件 (垃圾邮件概率接近阈值)")
        
        return result

def main():
    """主函数：运行完整的TextCNN模型实验"""
    print("="*60)
    print("TextCNN垃圾邮件检测系统 - F0.3评分标准")
    print("="*60)
    
    # 1. 初始化分类器
    classifier = TextCNN_SpamClassifier(
        random_state=RANDOM_STATE, 
        confidence_threshold=0.5,
        max_features=5000,      # 最大词汇量
        max_len=100,            # 文本最大长度
        embedding_dim=100,      # 词向量维度
        filters=128,            # 卷积核数量
        kernel_sizes=[3, 4, 5], # 不同尺寸的卷积核
        dropout_rate=0.5        # Dropout比率
    )
    
    # 2. 加载和准备数据
    df = classifier.load_and_prepare_data('spam mail.csv')
    
    if df is None or len(df) == 0:
        print("无法加载数据，程序退出")
        return
    
    # 3. 划分数据集 (8:2比例)
    X_train, X_test, y_train, y_test = classifier.prepare_data(df, test_size=0.2)
    
    if X_train is None:
        print("数据划分失败，程序退出")
        return
    
    # 4. 训练TextCNN模型
    history = classifier.train_model(
        X_train, y_train,
        epochs=20,              # 训练轮数
        batch_size=32,          # 批大小
        validation_split=0.1    # 验证集比例
    )
    
    # 5. 绘制训练历史
    print("\n" + "="*60)
    print("训练历史可视化")
    print("="*60)
    classifier.plot_training_history()
    
    # 6. 在默认阈值(0.5)下评估模型性能
    print("\n" + "="*60)
    print("使用默认阈值(0.5)的模型性能")
    print("="*60)
    metrics_default = classifier.evaluate_model(X_test, y_test, threshold=0.5)
    
    # 7. 寻找使F0.3分数最大的最佳阈值
    best_threshold, best_metrics = classifier.find_best_f03_threshold(
        X_test, y_test, 
        threshold_start=0.5, 
        threshold_step=0.01, 
        max_threshold=0.99
    )
    
    # 8. 使用最佳阈值评估模型性能
    print("\n" + "="*60)
    print(f"使用最佳阈值({best_threshold:.2f})的模型性能")
    print("="*60)
    final_metrics = classifier.evaluate_model(X_test, y_test, threshold=best_threshold)
    
    # 9. 绘制置信度分布
    print("\n" + "="*60)
    print("置信度分布可视化")
    print("="*60)
    classifier.plot_confidence_distribution(X_test, y_test, threshold=best_threshold)
    
    # 10. 示例预测
    print("\n" + "="*60)
    print("示例预测")
    print("="*60)
    
    # 示例1: 正常邮件
    print("\n示例1 - 正常邮件:")
    normal_text = "Hi team, please find attached the meeting schedule for next week. Let me know if you have any conflicts."
    classifier.predict_example(normal_text, threshold=best_threshold)
    
    # 示例2: 垃圾邮件
    print("\n示例2 - 垃圾邮件:")
    spam_text = "Congratulations! You've been selected to receive a FREE iPhone. Click here to claim your prize now! Limited time offer!"
    classifier.predict_example(spam_text, threshold=best_threshold)
    
    # 示例3: 模糊邮件
    print("\n示例3 - 模糊邮件:")
    ambiguous_text = "Dear user, we need to verify your account information. Please click the link below to confirm your details."
    classifier.predict_example(ambiguous_text, threshold=best_threshold)
    
    # 11. 模型参数总结
    print("\n" + "="*60)
    print("模型参数总结")
    print("="*60)
    
    print("TextCNN模型参数:")
    print(f"  最大词汇量: {classifier.max_features}")
    print(f"  最大序列长度: {classifier.max_len}")
    print(f"  词向量维度: {classifier.embedding_dim}")
    print(f"  卷积核数量: {classifier.filters}")
    print(f"  卷积核大小: {classifier.kernel_sizes}")
    print(f"  Dropout比率: {classifier.dropout_rate}")
    print(f"  最佳阈值: {best_threshold:.2f}")
    
    # 12. 性能对比总结
    print("\n" + "="*60)
    print("性能对比总结 (F0.3评分标准)")
    print("="*60)
    
    # 计算混淆矩阵详情
    cm_default = metrics_default['confusion_matrix']
    tn_default, fp_default, fn_default, tp_default = cm_default.ravel()
    fpr_default = fp_default / (fp_default + tn_default) if (fp_default + tn_default) > 0 else 0
    
    cm_final = final_metrics['confusion_matrix']
    tn_final, fp_final, fn_final, tp_final = cm_final.ravel()
    fpr_final = fp_final / (fp_final + tn_final) if (fp_final + tn_final) > 0 else 0
    
    comparison_df = pd.DataFrame({
        '指标': ['准确率', '精确率', '召回率', 'F1分数', 'F0.3分数', '误报率(FPR)'],
        '阈值=0.50': [
            f"{metrics_default['accuracy']:.4f}",
            f"{metrics_default['precision']:.4f}",
            f"{metrics_default['recall']:.4f}",
            f"{metrics_default['f1']:.4f}",
            f"{metrics_default['f03']:.4f}",
            f"{fpr_default:.4f}"
        ],
        f'阈值={best_threshold:.2f}': [
            f"{final_metrics['accuracy']:.4f}",
            f"{final_metrics['precision']:.4f}",
            f"{final_metrics['recall']:.4f}",
            f"{final_metrics['f1']:.4f}",
            f"{final_metrics['f03']:.4f}",
            f"{fpr_final:.4f}"
        ],
        '变化': [
            f"{final_metrics['accuracy'] - metrics_default['accuracy']:+.4f}",
            f"{final_metrics['precision'] - metrics_default['precision']:+.4f}",
            f"{final_metrics['recall'] - metrics_default['recall']:+.4f}",
            f"{final_metrics['f1'] - metrics_default['f1']:+.4f}",
            f"{final_metrics['f03'] - metrics_default['f03']:+.4f}",
            f"{fpr_final - fpr_default:+.4f}"
        ]
    })
    
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)
    print("总结:")
    print(f"- 使用TextCNN进行垃圾邮件检测")
    print(f"- 评分标准: F0.3分数 (更重视精确率)")
    print(f"- 最佳阈值: {best_threshold:.2f}")
    print(f"- 测试集F0.3分数: {final_metrics['f03']:.4f}")
    print(f"- 误报率(FPR): {fpr_final:.4f} (正常邮件被误判的比例)")
    print(f"- ROC曲线AUC面积: {auc(roc_curve(y_test, final_metrics['y_pred_proba'])[0], roc_curve(y_test, final_metrics['y_pred_proba'])[1]):.4f}")
    
    return classifier, final_metrics, best_threshold

# 运行主程序
if __name__ == "__main__":
    try:
        # 检查GPU是否可用
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        if gpu_available:
            print("检测到GPU，将使用GPU加速训练")
        else:
            print("未检测到GPU，将使用CPU进行训练")
        
        classifier, metrics, best_threshold = main()
    except FileNotFoundError:
        print("错误: 未找到 'spam mail.csv' 文件")
        print("请确保数据集文件在当前目录中")
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()