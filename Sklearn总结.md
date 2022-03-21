# 特征提取与处理

## 总结

CountVectorizer 类会把文档全部转换成小写，通过正则表达式用空格分割句子，然后抽取长度大于等于2的字母序列，即文档词块化（tokenize）

- **文档词块化**：把句子分割成词块（token）或有意义的字母序列的过程
- **词块**：大多是单词，也可能是一些短语，如标点符号和词缀

CountVectorizer 类可以通过设置 stop_words 参数过滤停用词，默认是英语常用的停用词 stop_words="english"

- **停用词（Stop-word）**：通常是构建文档意思的功能词汇，其字面意义并不体现，例如 a，an，the，do，be，will，on，around，beneath等
- **词根还原（stemming）**：把附加的词缀都去掉，构成一个词块，可能不是一个正常的单词，例如every的词根是everi。通常需要用规则产生词干（stem）并操作词块，不需要词法资源，也不在乎单词的意思
- **词形还原（lemmatization）**：用来处理可以表现单词意思的词元（lemma）或形态学的词根（morphological root）的过程。词元是单词在词典中查询该词的基本形式，例如was的词元为be。词形还原通常需要词法资源的支持，比如WordNet

## 示例

```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize, pos_tag
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


def tokenize(corpus):
    # 词快化：nltk.word_tokenize(document)
    return [word_tokenize(document) for document in corpus]


def tokenize_with_tag(corpus):
    # 词快化并标注：nltk.word_tokenize(document)
    return [pos_tag(word_tokenize(document)) for document in corpus]


def stem(token):
    # 单个token词根还原：nltk.PorterStemmer().stem(token)
    return PorterStemmer().stem(token)


def lemmatize(token, tag_lower_init):
    # 单个token词形还原：nltk.stem.wordnet.WordNetLemmatizer().lemmatize(token, tag_lower_init)
    return WordNetLemmatizer().lemmatize(token, tag_lower_init)


def stemming(corpus):
    # 整个语料库词根还原
    return [" ".join([stem(token) for token in tokend_document]) for tokend_document in tokenize(corpus)]


def lemmatization(coprpus, tag_lower_init_list=None):
    # 整个语料库词形还原
    return [" ".join([lemmatize(token, tag[0].lower()) if tag[0].lower() in tag_lower_init_list else token
                      for token, tag in tokend_document_with_tag]) for tokend_document_with_tag in tokenize_with_tag(corpus)]


print("----------------分类变量特征提取----------------")
# 独热码形式
onehot_encoder = DictVectorizer()
instances = [{"city": "New York"}, {
    "city": "San Francisco"}, {"city": "Chapel Hill"}]
print("独热编码结果：\n", onehot_encoder.fit_transform(instances).toarray())


print("----------------文字特征提取----------------")
# 语料库（多篇文档组成，以下以4个文档为例）
corpus = [
    "He ate the sandwiches",
    "Every sandwich was eaten by him",
    "Duke lost lost a basketball",
]
print("原始语料库：\n", corpus)


print("--------基本用法--------")
vectorizer_1 = CountVectorizer()
print("词数编码：\n", vectorizer_1.fit_transform(corpus).todense())  # 词块化结果
print("词汇表：\n", vectorizer_1.vocabulary_)  # 词汇表，即词块索引表


print("--------使用停用词过滤--------")
vectorizer_2 = CountVectorizer(stop_words="english")
print("词数编码：\n", vectorizer_2.fit_transform(corpus).todense())
print("词汇表：\n", vectorizer_2.vocabulary_)


print("--------使用词根还原--------")
stemmed_corpus = stemming(corpus)
print("词根还原后语料库：\n", stemmed_corpus)
vectorizer_3 = CountVectorizer()
print("词数编码：\n", vectorizer_3.fit_transform(stemmed_corpus).todense())
print("词汇表：\n", vectorizer_3.vocabulary_)


print("--------使用词形还原--------")
lemmatized_corpus = lemmatization(corpus, ["n", "v"])
print("词形还原后语料库：\n", lemmatized_corpus)
vectorizer_4 = CountVectorizer()
print("词数编码：\n", vectorizer_4.fit_transform(lemmatized_corpus).todense())
print("词汇表：\n", vectorizer_4.vocabulary_)


print("--------TF-IDF--------")
vectorizer_5 = TfidfVectorizer(stop_words="english")
print("TF-IDF：\n", vectorizer_5.fit_transform(corpus).todense())
print("词汇表：\n", vectorizer_5.vocabulary_)
```

----------------分类变量特征提取----------------
独热编码结果：
[[0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]]
----------------文字特征提取----------------
原始语料库：
['He ate the sandwiches', 'Every sandwich was eaten by him', 'Duke lost lost a basketball']
--------基本用法--------
词数编码：
[[1 0 0 0 0 0 1 0 0 0 1 1 0]
 [0 0 1 0 1 1 0 1 0 1 0 0 1]
 [0 1 0 1 0 0 0 0 2 0 0 0 0]]
词汇表：
{'he': 6, 'ate': 0, 'the': 11, 'sandwiches': 10, 'every': 5, 'sandwich': 9, 'was': 12, 'eaten': 4, 'by': 2, 'him': 7, 'duke': 3, 'lost': 8, 'basketball': 1}
--------使用停用词过滤--------
词数编码：
[[1 0 0 0 0 0 1]
 [0 0 0 1 0 1 0]
 [0 1 1 0 2 0 0]]
词汇表：
{'ate': 0, 'sandwiches': 6, 'sandwich': 5, 'eaten': 3, 'duke': 2, 'lost': 4, 'basketball': 1}
--------使用词根还原--------
词根还原后语料库：
['he ate the sandwich', 'everi sandwich wa eaten by him', 'duke lost lost a basketbal']
词数编码：
[[1 0 0 0 0 0 1 0 0 1 1 0]
 [0 0 1 0 1 1 0 1 0 1 0 1]
 [0 1 0 1 0 0 0 0 2 0 0 0]]
词汇表：
{'he': 6, 'ate': 0, 'the': 10, 'sandwich': 9, 'everi': 5, 'wa': 11, 'eaten': 4, 'by': 2, 'him': 7, 'duke': 3, 'lost': 8, 'basketbal': 1}
--------使用词形还原--------
词形还原后语料库：
['He eat the sandwich', 'Every sandwich be eat by him', 'Duke lose lose a basketball']
词数编码：
[[0 0 0 0 1 0 1 0 0 1 1]
 [0 1 1 0 1 1 0 1 0 1 0]
 [1 0 0 1 0 0 0 0 2 0 0]]
词汇表：
{'he': 6, 'eat': 4, 'the': 10, 'sandwich': 9, 'every': 5, 'be': 1, 'by': 2, 'him': 7, 'duke': 3, 'lose': 8, 'basketball': 0}
--------TF-IDF--------
TF-IDF：
[[0.70710678             0.                          0.                         0.                          0.                           0.                          0.70710678]
 [0.                                0.                          0.                         0.70710678         0.                          0.70710678         0.                 ]
 [0.                                0.40824829        0.40824829        0.                          0.81649658         0.                          1.                 ]]
词汇表：
{'ate': 0, 'sandwiches': 6, 'sandwich': 5, 'eaten': 3, 'duke': 2, 'lost': 4, 'basketball': 1}



# 加载数据集

## 总结

#### 波士顿房价：回归

```python
from sklearn.datasets import load_boston
data = load_boston()
X, y =data.data, data.target.reshape(-1, 1) # X：shape=[506, 13]，y：shape=[506, 1]
```

#### 手写数字：10类

```python
from sklearn.datasets import load_digits
data = load_digits()
X = data.data 或 X = data.images # 区别：前者shape=[1797, 64]，后者[1797, 8, 8]
y = data.target.reshape(-1, 1) # shape=[1797, 1] 
```

#### 纸鸢花：3类

 ```python
 from sklearn.datasets import load_iris
 data = load_iris()
 X, y = data.data, data.target.reshape(-1, 1) # X：shape=[150, 4]，y：shape=[150, 1]
 ```



# 训练集划分

## 总结

 ```python
 from sklearn. model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y)
 ```



# 标准化

## 总结

 ```python
 from sklearn.preprocessing import StandardScaler
 SS_X = StandardScaler()
 SS_y = StandardScaler()
 X_train = SS_X.fit_transform(X_train)
 X_test = SS_X.transform(X_test)
 y_train = SS_y.fit_transform(y_train)
 y_test = SS_y.transform(y_test)
 # fit_transform和transform的区别在于前者能更新mean和var，而后者不能
 
 # 查看均值，方差：
 SS_X.mean, SS_X.var
 SS_y.mean, SS_y.var
 ```



# 网格搜索最优超参数

## 总结

 ```python
 # 以逻辑回归为例：
 from sklearn.model_selection import train_test_split, GridSearchCV
 from sklearn.pipeline import Pipeline
 pipeline = Pipeline([("clf", LogisticRegression())])
 parameters = {"clfpenalty": ("l1", "l2"), "clfC": (0.01, 0.1, 1, 10)}
 grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0, scoring="accuracy", cv=3)
 grid_search.fit(X_train, y_train)
 best_parameters = grid_search.best_estimator_.get_params()
 best_score = grid_search.best_score_
 print("最佳效果：%0.3f" % best_score)
 print("最优参数组合：", [[para_name, best_parameters[para_name]] for para_name in parameters.keys()])
 ```

 

# 保存和加载模型

## 总结

 ```python
 import joblib
 joblib.dump(model, 'model.pkl') # 模型保存
 model = joblib.load('model.pkl') # 模型加载
 ```



# 模型评估

## 总结

#### 评估score

 ```python
 model.score(X_test, y_test)
 sklearn.linear_model.LinearRegression：# 默认r2_score 
 sklearn.linear_model.SGDRegressor：# 默认为”squared_loss”
 sklearn.linear_model.LogisticRegression：# 默认为”accuracy”
 sklearn.tree.DecisionTreeClassifier：# 默认为”accuracy”
 ```

#### 交叉验证

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring=None, cv=k)
# scoring=None默认使用model的评估方法，可设为”squared_loss”、”accuracy”、”precision”、”recall”、”f1”等
```

#### 混淆矩阵

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

#### 准确率

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```

#### 精确率

```python
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
```

#### 召回率

```python
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
```

#### F1

```python
from sklearn.metrics import f1_score
f1 = recall_score(y_test, y_pred)
```

#### ROC

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 0])
```

#### AUC

```python
from sklearn.metrics import auc
auc_val = auc(fpr, tpr)
```

#### 打印报告

```python
from sklearn.metrics import classification_report
classification_report(y_test, y_pred)
```



# 线性回归

## 总结

#### 普通线性回归

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.predict(X_test)
```

#### 多项式回归 (k) 阶

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
PF = PolynomialFeatures(degree=k)
X_train_k, y_train_k = PF.fit_transform(X_train), y_train
X_test_k, y_test_k = PF.transform(X_test), y_test
model = LinearRegression()
model.fit(X_train_k, y_train_k)
model.predict(X_test_k)
```

## 示例

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

print("----------------普通线性回归----------------")
# 训练集
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

# 测试集
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

# 创建并拟合模型
model = LinearRegression()
model.fit(X_train, y_train)

# 模型评估
R2_score = model.score(X_test, y_test)
print("普通线性回归的R方：", R2_score)

# 预测
print("普通线性回归的预测：\n", model.predict(X_test))


print("----------------多项式回归（2阶）----------------")
# 处理数据集
PF_2 = PolynomialFeatures(degree=2)
X_train_2, y_train_2 = PF_2.fit_transform(X_train), y_train
X_test_2, y_test_2 = PF_2.transform(X_test), y_test

# 创建并拟合模型
model_2 = LinearRegression()
model_2.fit(X_train_2, y_train_2)

# 模型评估
R2_score_2 = model_2.score(X_test_2, y_test_2)
print("多项式回归（2阶）的R方：", R2_score_2)

# 预测
print("多项式回归（2阶）的预测：\n", model_2.predict(X_test_2))


print("----------------多项式回归（3阶）----------------")
# 处理数据集
PF_3 = PolynomialFeatures(degree=3)
X_train_3, y_train_3 = PF_3.fit_transform(X_train), y_train
X_test_3, y_test_3 = PF_3.transform(X_test), y_test

# 创建并拟合模型
model_3 = LinearRegression()
model_3.fit(X_train_3, y_train_3)

# 模型评估
R2_score_3 = model_3.score(X_test_3, y_test_3)
print("多项式回归（3阶）的R方：", R2_score_3)

# 预测
print("多项式回归（3阶）的预测：\n", model_3.predict(X_test_3))


print("----------------多项式回归（7阶）----------------")
# 处理数据集
PF_7 = PolynomialFeatures(degree=7)
X_train_7, y_train_7 = PF_7.fit_transform(X_train), y_train
X_test_7, y_test_7 = PF_7.transform(X_test), y_test

# 创建并拟合模型
model_7 = LinearRegression()
model_7.fit(X_train_7, y_train_7)

# 模型评估
R2_score_7 = model_7.score(X_test_7, y_test_7)
print("多项式回归（7阶）的R方：", R2_score_7)

# 预测
print("多项式回归（7阶）的预测：\n", model_7.predict(X_test_7))


"""----------------绘制结果----------------"""
plt.figure(figsize=(6, 4))
plt.rcParams["font.sans-serif"], plt.rcParams["axes.unicode_minus"] = ["SimHei"], False
plt.plot(X_train, y_train, "k.", label="训练样本")
xx = np.linspace(0, 26, 100).reshape(-1, 1)
plt.plot(xx, model.predict(xx), "b", label="普通线性回归拟合曲线")
plt.plot(xx, model_2.predict(PF_2.fit_transform(xx)),
         "r", label="多项式回归（2阶）拟合曲线")
plt.plot(xx, model_3.predict(PF_3.fit_transform(xx)),
         "g", label="多项式回归（3阶）拟合曲线")
plt.plot(xx, model_7.predict(PF_7.fit_transform(xx)),
         "c", label="多项式回归（7阶）拟合曲线")
plt.title("线性回归")
plt.xlabel("X")
plt.ylabel("y")
plt.axis([0, 25, 0, 25])
plt.legend()
plt.grid(True)
plt.show()
```

-----------------普通线性回归----------------
普通线性回归的R方： 0.809726797707665
普通线性回归的预测：
[[ 7.82327586]
 [ 9.77586207]
 [12.70474138]
 [17.5862069 ]]
----------------多项式回归（2阶）----------------
多项式回归（2阶）的R方： 0.8675443656345054
多项式回归（2阶）的预测：
[[ 6.38646055]
 [10.0021322 ]
 [14.19529584]
 [17.90298507]]
----------------多项式回归（3阶）----------------
多项式回归（3阶）的R方： 0.8356924156037133
多项式回归（3阶）的预测：
[[ 6.84492481]
 [ 9.4962406 ]
 [14.03374648]
 [18.67763158]]
----------------多项式回归（7阶）----------------
多项式回归（7阶）的R方： 0.49198460568667457
多项式回归（7阶）的预测：
[[ 7.        ]
 [ 9.        ]
 [15.22030267]
 [13.78510859]]

<img src="assets/output_3_1.png" alt="output_3_1"  />



# 梯度下降法拟合

## 总结

 ```python
 from sklearn.linear_model import SGDRegressor
 model = SGDRegressor(loss="squared_loss") # 默认是squared_loss
 model.fit(X_train, y_train)
 model.predict(X_test)
 ```

## 示例

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 波士顿房价数据集
data = load_boston()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))


# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 标准化处理
SS_X = StandardScaler()
SS_y = StandardScaler()
X_train = SS_X.fit_transform(X_train)
X_test = SS_X.transform(X_test)
y_train = SS_y.fit_transform(y_train)
y_test = SS_y.transform(y_test)

# 创建并拟合模型
model = SGDRegressor(loss="squared_loss")  # 默认是squared_loss
model.fit(X_train, y_train)

# 模型评估
# 训练集上交叉验证
scores = cross_val_score(model, X_train, y_train, cv=5)
print('交叉验证R方:', scores)
print('交叉验证R方的均值:', np.mean(scores))
# 测试集
print('测试集R方:', model.score(X_test, y_test))
```

X的形状：(506, 13) 	           y的形状：(506, 1)
X_train的形状：(379, 13) 	 X_test的形状：(127, 13) 	 y_train的形状：(379, 1) 	 y_test的形状：(127, 1)
交叉验证R方: [0.61452695 0.61025864 0.7385156  0.75706177 0.6904955 ]
交叉验证R方的均值: 0.6821716924586292
测试集R方: 0.7309539016823225



# 逻辑回归

## 总结

 ```python
 from sklearn.linear_model import LogisticRegression
 model = LinearRegression()
 model.fit(X_train, y_train)
 model.predict(X_test) 或 model.predict_proba(X_test)
 ```

## 示例

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 读取和预处理数据
data = pd.read_csv("datasets/KaggleCredit2.csv", index_col=0, header=0)
print("原始数据集data的形状：", data.shape)
data.dropna(inplace=True)   # 去掉为空的数据
print("去掉空值后数据集data的形状：", data.shape)
X = np.array(data.drop("SeriousDlqin2yrs", axis=1))  # 选择SeriousDlqin2yrs之外的特征
y = np.array(data["SeriousDlqin2yrs"]).reshape(-1, 1)  # 选择SeriousDlqin2yrs特征
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 网格搜索最优超参数
"""
pipeline = Pipeline([("clf", LogisticRegression())])
parameters = {"clf__penalty": ("l1", "l2"), "clf__C": (0.01, 0.1, 1, 10)}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0, scoring="accuracy", cv=3)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
best_score = grid_search.best_score_
print("最佳效果：%0.3f" % best_score)
print("最优参数组合：", [[para_name, best_parameters[para_name]] for para_name in parameters.keys()])
"""

# 创建并拟合模型
model = LogisticRegression(penalty="l2", C=0.01)
model.fit(X_train, y_train)

# 预测
y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)

# 混淆矩阵
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
# 准确率
accuracy_train, accuracy_test = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
# 精确率
precision_train, precision_test = precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)
# 召回率
recall_train, recall_test = recall_score(y_train, y_train_pred,), recall_score(y_test, y_test_pred)
# F1
f1_train, f1_test = f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)
# ROC
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, model.predict_proba(X_train)[:, 0])
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, model.predict_proba(X_test)[:, 0])
# AUC
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

print("训练集混淆矩阵：\n", cm_train)
print("测试集混淆矩阵：\n", cm_test)
print("训练集准确率：{} \t 测试集准确率：{}".format(accuracy_train, accuracy_test))
print("训练集精确率：{} \t 测试集精确率：{}".format(precision_train, precision_test))
print("训练集召回率：{} \t 测试集召回率：{}".format(recall_train, recall_test))
print("训练集F1：{} \t 测试集F1：{}".format(f1_train, f1_test))
print("训练集AUC：{} \t 测试集AUC：{}".format(auc_train, auc_test))

plt.rcParams["font.sans-serif"], plt.rcParams["axes.unicode_minus"] = ["SimHei"], False
plt.title('ROC')
plt.plot(fpr_train, tpr_train, "b")
plt.plot(fpr_test, tpr_test, "r")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
```

原始数据集data的形状： (112915, 11)
去掉空值后数据集data的形状： (108648, 11)
X的形状：(108648, 10) 	         y的形状：(108648, 1)
X_train的形状：(81486, 10) 	 X_test的形状：(27162, 10) 	 y_train的形状：(81486, 1) 	 y_test的形状：(27162, 1)
训练集混淆矩阵：
[[75925    96]
 [ 5362   103]]
测试集混淆矩阵：
[[25265    36]
 [ 1819    42]]
训练集准确率：0.9330191689369953 	     测试集准确率：0.9317060599366762
训练集精确率：0.5175879396984925 	     测试集精确率：0.5384615384615384
训练集召回率：0.018847209515096067 	 测试集召回率：0.022568511552928532
训练集F1：0.03637005649717514 	          测试集F1：0.043321299638989175
训练集AUC：0.3146387982816854 	         测试集AUC：0.31765572597277514

![output_7_1](assets/output_7_1.png)



# SVM

## 总结

 ```python
 from sklearn.svm import SVC
 model = SVC(kernel="rbf", C=2.0, gamma=0.1)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 ```

## 示例

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 纸鸢花数据集，有3类
data = load_iris()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 创建并拟合模型
model = SVC(kernel="rbf", C=2.0, gamma=0.1)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("预测评估：\n", classification_report(y_test, y_pred))
```

X的形状：(150, 4) 	           y的形状：(150, 1)
X_train的形状：(112, 4) 	 X_test的形状：(38, 4) 	 y_train的形状：(112, 1) 	 y_test的形状：(38, 1)
预测评估：
                      precision   recall  f1-score   support

​              0              1.00      1.00      1.00        18
​              1              1.00      0.93      0.96        14
​              2              0.86      1.00      0.92         6

​        accuracy                                  0.97        38

​     macro avg       0.95      0.98      0.96        38
weighted avg       0.98      0.97      0.97        38



# 决策树

## 总结

 ```python
 from sklearn.tree import DecisionTreeClassifier
 model = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=2, min_samples_leaf=3)
 model.fit(X_train, y_train)
 model.predict(X_test)
 ```

## 示例

```python
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from IPython.display import display, Image
import pydotplus
import numpy as np
import pandas as pd

data = pd.read_csv("datasets/DecisionTree.csv", index_col=None, header=0)
print("原始数据集data的形状：", data.shape)
data.dropna(inplace=True)   # 去掉为空的数据
print("去掉空值后数据集data的形状：", data.shape)
# 选择income之外的特征，并利用pd.get_dummies实现每个特征的独热编码
X_df = pd.get_dummies(data.drop("income", axis=1))
X, feature_names = np.array(X_df), X_df.columns
# 选择income特征
y = np.array(data["income"]).reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 网格搜索最优超参数
"""
pipeline = Pipeline([("clf", DecisionTreeClassifier(criterion="entropy"))])
parameters = {"clf__max_depth": (9, 10, 11),
              "clf__min_samples_split": (2, 3),
              "clf__min_samples_leaf": (1, 2, 3)}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0, scoring="accuracy")
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
best_score = grid_search.best_score_
print("最佳效果：%0.3f" % best_score)
print("最优参数组合：", [[para_name, best_parameters[para_name]] for para_name in parameters.keys()])
"""

# 创建并拟合模型
model = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=2, min_samples_leaf=3)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("预测评估：\n", classification_report(y_test, y_pred))

# 可视化决策树
dot_data = export_graphviz(model, out_file=None, feature_names=feature_names,
                           class_names=["<=50k", ">50k"], filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))
```

原始数据集data的形状： (32561, 9)
去掉空值后数据集data的形状： (32561, 9)
X的形状：(32561, 102) 	           y的形状：(32561, 1)
X_train的形状：(24420, 102) 	 X_test的形状：(8141, 102) 	 y_train的形状：(24420, 1) 	 y_test的形状：(8141, 1)
预测评估：
                         precision  recall  f1-score   support

​         <=50K          0.85      0.93      0.89      6170
​           \>50K          0.69      0.48      0.57      1971

​        accuracy                                  0.82      8141

​     macro avg       0.77      0.71      0.73      8141
weighted avg       0.81      0.82      0.81      8141

dot: graph is too large for cairo-renderer bitmaps. Scaling by 0.868069 to fit

![output_9_1](assets/output_9_1.png)



# 随机森林

## 总结

 ```python
 from sklearn.ensemble import RandomForestClassifier
 model = RandomForestClassifier(criterion="entropy", n_estimators=40, max_depth=16, min_samples_split=2, min_samples_leaf=1)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 ```

## 示例

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from IPython.display import display, Image
import pydotplus
import numpy as np
import pandas as pd

data = pd.read_csv("datasets/DecisionTree.csv", index_col=None, header=0)
print("原始数据集data的形状：", data.shape)
data.dropna(inplace=True)   # 去掉为空的数据
print("去掉空值后数据集data的形状：", data.shape)
# 选择income之外的特征，并利用pd.get_dummies实现每个特征的独热编码
X_df = pd.get_dummies(data.drop("income", axis=1))
X, feature_names = np.array(X_df), X_df.columns
# 选择income特征
y = np.array(data["income"]).reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 网格搜索最优超参数
"""
pipeline = Pipeline([("clf", RandomForestClassifier(criterion="entropy", min_samples_split=2, min_samples_leaf=1))])
parameters = {"clf__n_estimators": (30, 35, 40), "clf__max_depth": (16, 17, 18)}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0, scoring="accuracy")
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
best_score = grid_search.best_score_
print("最佳效果：%0.3f" % best_score)
print("最优参数组合：", [[para_name, best_parameters[para_name]] for para_name in parameters.keys()])
"""

# 创建并拟合模型
model = RandomForestClassifier(criterion="entropy", n_estimators=40, max_depth=16, min_samples_split=2, min_samples_leaf=1)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("预测评估：\n", classification_report(y_test, y_pred))
```

原始数据集data的形状： (32561, 9)
去掉空值后数据集data的形状： (32561, 9)
X的形状：(32561, 102) 	           y的形状：(32561, 1)
X_train的形状：(24420, 102) 	 X_test的形状：(8141, 102) 	 y_train的形状：(24420, 1) 	 y_test的形状：(8141, 1)
预测评估：
                        precision  recall  f1-score   support

​        <=50K           0.86      0.94      0.90      6193
​          \>50K           0.72      0.50      0.59      1948

​       accuracy                                   0.83      8141

​     macro avg       0.79      0.72      0.74      8141
weighted avg       0.82      0.83      0.82      8141



# GBDT

## 总结

 ```python
 from sklearn.ensemble import GradientBoostingRegressor
 model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 ```

##  示例

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 波士顿房价数据集
data = load_boston()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 创建并拟合模型
model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("测试集上R2：\n", r2_score(y_test, y_pred))
```

X的形状：(506, 13) 	           y的形状：(506, 1)
X_train的形状：(379, 13) 	 X_test的形状：(127, 13) 	 y_train的形状：(379, 1) 	 y_test的形状：(127, 1)
测试集上R2：
0.8684985983424158



# XGBoost

## 总结

 ```python
 from xgboost import XGBRegressor
 model = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=3)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 ```

## 示例

```python
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 波士顿房价数据集
data = load_boston()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 创建并拟合模型
model = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=3)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("测试集上R2：\n", r2_score(y_test, y_pred))
```

X的形状：(506, 13) 	           y的形状：(506, 1)
X_train的形状：(379, 13) 	 X_test的形状：(127, 13) 	 y_train的形状：(379, 1) 	 y_test的形状：(127, 1)
测试集上R2：
0.8950043788558586



# LightGBM

## 总结

 ```python
 from lightgbm import LGBMRegressor
 model = LGBMRegressor(boosting_type="gbdt", learning_rate=0.1, n_estimators=100, max_depth=3)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 ```

## 示例

```python
from lightgbm import LGBMRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 波士顿房价数据集
data = load_boston()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 创建并拟合模型
model = LGBMRegressor(boosting_type="gbdt", learning_rate=0.1, n_estimators=100, max_depth=3)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("测试集上R2：\n", r2_score(y_test, y_pred))
```

X的形状：(506, 13) 	           y的形状：(506, 1)
X_train的形状：(379, 13) 	 X_test的形状：(127, 13) 	 y_train的形状：(379, 1) 	 y_test的形状：(127, 1)
测试集上R2：
0.8898112666074992



# KNN

## 总结

 ```python
 from sklearn.neighbors import KNeighborsClassifier
 model = KNeighborsClassifier(n_neighbors=3)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 ```

## 示例

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 纸鸢花数据集，有3类
data = load_iris()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 创建并拟合模型
k = 3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("预测评估：\n", classification_report(y_test, y_pred))
```

X的形状：(150, 4) 	           y的形状：(150, 1)
X_train的形状：(112, 4) 	 X_test的形状：(38, 4) 	 y_train的形状：(112, 1) 	 y_test的形状：(38, 1)
预测评估：
                        precision   recall  f1-score   support

​               0             1.00      1.00      1.00        14
​               1             1.00      0.92      0.96        12
​               2             0.92      1.00      0.96        12

​        accuracy                                  0.97        38

​     macro avg       0.97      0.97      0.97        38
weighted avg       0.98      0.97      0.97        38



# 聚类

## 总结

#### K-means

 ```python
 from sklearn.cluster import KMeans
 model = KMeans(n_clusters=k)
 model.fit(X)
 labels, centers = model.labels, model.cluster_centers
 ```

#### 高斯混合模型GMM

```python
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=k).fit(X)
labels = model.predict(X)
probas = np.max(model.predict_proba(X),axis=1).round(3)
```

## 示例

```python
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt


def best_kmeans(X, kmin=1, kmax=10):
    k_range = range(kmin, kmax+1)
    meanDistortions = []
    for k in k_range:
        model = KMeans(n_clusters=k)
        model.fit(X)
        # X.shape=(m,2)  m是数据量，2是特征数
        # kmeans.cluster_centers_.shape=(k,2) # k是聚类数
        # cdist(X, kmeans.cluster_centers_, "euclidean").shape=(m,k) 计算每个数据到每个聚类中心的聚类
        meanDistortions.append(
            np.mean(np.min(cdist(X, model.cluster_centers_, "euclidean"), axis=1)))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    plt.rcParams['font.sans-serif'], plt.rcParams['axes.unicode_minus'] = ['SimHei'], False
    ax[0].plot(X[:, 0], X[:, 1], "k.")
    ax[0].grid(True)
    ax[0].set_title("原始数据")
    ax[1].plot(k_range, meanDistortions, "bx-")
    ax[1].set_title("用肘部法则来确定最佳的K值")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("平均类内距离平方和")
    ax[1].axis([kmin, kmax, 0, max(meanDistortions)])
    plt.show()


# 随机生成数据
cluster1 = np.random.uniform(0.5, 1.5, (10, 2))
cluster2 = np.random.uniform(3.5, 4.5, (10, 2))
cluster3 = np.random.uniform(6.5, 7.5, (10, 2))
X = np.vstack((cluster1, cluster2, cluster3))  # shape=(30,2)

print("----------------K-means----------------")

# 确定最佳聚类数
best_kmeans(X, kmin=1, kmax=8)

# 创建并拟合模型
k = 3
model = KMeans(n_clusters=k)
model.fit(X)

# 打印结果
labels, centers = model.labels_, model.cluster_centers_
print("聚类结果：", labels)
print("聚类中心：\n", centers)
print("轮廓系数：", silhouette_score(X, labels, metric="euclidean"))

# 绘制聚类结果
plt.figure(figsize=(5, 3))
for i in range(k):
    cluster = X[np.where(labels == i)[0]]
    plt.plot(cluster[:, 0], cluster[:, 1], ".", label="簇 %d" % i)
plt.plot(centers[:, 0], centers[:, 1], "x", label="聚类中心")
plt.title("聚类结果")
plt.grid(True)
plt.legend()
plt.show()

print("----------------GMM----------------")

# 创建并拟合模型
k = 3
model = GaussianMixture(n_components=k).fit(X)
labels = model.predict(X)
probas = np.max(model.predict_proba(X),axis=1).round(3)

# 打印结果
print("聚类结果：", labels)
print("归属概率：", probas)
print("轮廓系数：", silhouette_score(X, labels, metric="euclidean"))

# 绘制聚类结果
plt.figure(figsize=(5, 3))
for i in range(k):
    cluster = X[np.where(labels == i)[0]]
    plt.plot(cluster[:, 0], cluster[:, 1], ".", label="簇 %d" % i)
plt.title("聚类结果")
plt.grid(True)
plt.legend()
plt.show()
```

----------------K-means----------------

![output_13_1](assets/output_13_1.png)

聚类结果： [1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0]
聚类中心：
[[7.04552365 7.07002758]
 [1.05716823 0.9004473 ]
 [3.81740805 3.99756329]]
轮廓系数：0.8620903970475202

![output_13_3](assets/output_13_3.png)

----------------GMM----------------
聚类结果：[0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1]
归属概率：[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
轮廓系数：0.8620903970475202

![output_13_5](assets/output_13_5.png)



# PCA

## 总结

```python
from sklearn.decomposition import PCA
model = PCA(n_components=2)
Xr = model.fit_transform(X)
```

## 示例

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 纸鸢花数据集，有3类
data = load_iris()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 创建并拟合模型
model = PCA(n_components=2)
Xr = model.fit_transform(X)
print("X降维后的形状：", Xr.shape)

# 绘制PCA结果
plt.figure(figsize=(5, 3))
plt.scatter(Xr[:, 0], Xr[:, 1], c=y, s=10)
plt.title("使用PCA降至二维")
plt.grid(True)
plt.show()
```

X的形状：(150, 4) 	 y的形状：(150, 1)
X降维后的形状： (150, 2)

![output_15_1](assets/output_15_1.png)



#  朴素贝叶斯NB

## 总结

 ```python
 from sklearn.naive_bayes import MultinomialNB
 model = MultinomialNB()
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 ```

## 示例

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 纸鸢花数据集，有3类
data = load_iris()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 创建并拟合模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("预测评估：\n", classification_report(y_test, y_pred))
```

X的形状：(150, 4) 	           y的形状：(150, 1)
X_train的形状：(112, 4) 	 X_test的形状：(38, 4) 	 y_train的形状：(112, 1) 	 y_test的形状：(38, 1)
预测评估：
                        precision  recall  f1-score   support

​               0             1.00      1.00      1.00        15
​               1             0.62      1.00      0.77        10
​               2             1.00      0.54      0.70        13

​        accuracy                                  0.84        38

​     macro avg       0.88      0.85      0.82        38
weighted avg       0.90      0.84      0.84        38



# 感知机

## 总结

 ```python
 from sklearn.linear_model import Perceptron
 model = Perceptron(max_iter=1000, eta0=0.1)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 ```

## 示例

```python
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 纸鸢花数据集，有3类
data = load_iris()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 创建并拟合模型
model = Perceptron(max_iter=1000, eta0=0.1)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("预测评估：\n", classification_report(y_test, y_pred))
```

X的形状：(150, 4) 	           y的形状：(150, 1)
X_train的形状：(112, 4) 	 X_test的形状：(38, 4) 	 y_train的形状：(112, 1) 	 y_test的形状：(38, 1)
预测评估：
                       precision   recall  f1-score   support

​             0               0.84      1.00      0.91        16
​             1               0.60      0.50      0.55         6
​             2               1.00      0.88      0.93        16

​       accuracy                                   0.87        38

​     macro avg       0.81      0.79      0.80        38
weighted avg       0.87      0.87      0.86        38



# 多层感知机

## 总结

 ```python
 from sklearn.neural_network import MLPClassifier
 model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="logistic", solver="sgd",
                       batch_size=8, learning_rate_init=0.01, max_iter=200)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 ```

## 示例

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 纸鸢花数据集，有3类
data = load_iris()
X, y = data.data, data.target.reshape(-1, 1)
print("X的形状：{} \t y的形状：{}".format(X.shape, y.shape))

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train的形状：{} \t X_test的形状：{} \t y_train的形状：{} \t y_test的形状：{}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# 创建并拟合模型
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="logistic", solver="sgd",
                      batch_size=8, learning_rate_init=0.01, max_iter=200)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("预测评估：\n", classification_report(y_test, y_pred))
```

X的形状：(150, 4) 	           y的形状：(150, 1)
X_train的形状：(112, 4) 	 X_test的形状：(38, 4) 	 y_train的形状：(112, 1) 	 y_test的形状：(38, 1)
预测评估：
                        precision  recall  f1-score   support

​             0               1.00      1.00      1.00        15
​             1               1.00      1.00      1.00         8
​             2               1.00      1.00      1.00        15

​        accuracy                                  1.00        38

​     macro avg       1.00      1.00      1.00        38
weighted avg       1.00      1.00      1.00        38

