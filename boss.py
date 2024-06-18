import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 检查是否有GPU可用
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("No GPU available")

# 打印所有GPU设备信息
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("GPU Device ", gpu.name)


import pandas as pd



'''
数据导入及预处理
'''
base_dir="dataSet"
train_txt=os.path.join(base_dir,'train.txt')
test_txt=os.path.join(base_dir,'test.txt')
validation_txt=os.path.join(base_dir,'dev.txt')
class_txt=os.path.join(base_dir,'class.txt')



class_data=pd.read_csv(class_txt,sep=' ',names=['label_name','label'])

category_mapping = class_data.set_index('label')['label_name'].to_dict()

print(category_mapping)

train_data=pd.read_csv(train_txt,sep='\t',names=['content','label'])
test_data=pd.read_csv(test_txt,sep='\t',names=['content','label'])
validation_data=pd.read_csv(validation_txt,sep='\t',names=['content','label'])


print(train_data.shape)
print(test_data.shape)
print(validation_data.shape)


# print(train_data.info())

#特征工程
import jieba
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

# 不添加分词
#train_content = train_data['content']
#test_content = test_data['content']
#validation_content=validation_data['content']
# 添加分词

import numpy as np
train_content = train_data['content'].apply(chinese_word_cut)
test_content = test_data['content'].apply(chinese_word_cut)
validation_content=validation_data['content'].apply(chinese_word_cut)

print(type(train_content))
print(type(test_content))
print(type(validation_content))


train_content.to_csv("train_content.txt",index=False,header=False,sep=' ')
test_content.to_csv("test_content.txt",index=False,header=False,sep=' ')
validation_content.to_csv("validation_content.txt",index=False,header=False,sep=' ')



import sys
sys.exit(0)

'''

使用tf-idf模型构建

'''

# from sklearn.feature_extraction.text import TfidfVectorizer
# f_all = pd.concat(objs=[train_data['content'],validation_data['content'], test_data['content']], axis=0)
#
# tfidf_vect = TfidfVectorizer(max_df = 0.9,min_df = 3,token_pattern=r"(?u)\b\w+\b")
#
# tfidf_vect.fit(f_all)
#
# x_train=tfidf_vect.fit_transform(train_data['content'])
# x_validation=tfidf_vect.fit_transform(validation_data['content'])
#
# x_test=tfidf_vect.transform(test_data['content'])
#
#
# print(type(x_train))
# print(type(x_test))
# print(type(x_validation))

'''

LSTM的构建

'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Dropout,BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

'''
(180000, 2)
(10000, 2)
(10000, 2)
'''
# train_content =train_data['content'].apply(chinese_word_cut)
# test_content = test_data['content'].apply(chinese_word_cut)
# validation_content=validation_data['content'].apply(chinese_word_cut)

train_label=np.array(train_data['label'])
test_label=np.array(test_data['label'])
validation_label=np.array(validation_data['label'])



# 假设 texts 和 labels 分别是文本列表和对应的标签列表
# texts = ["这是一个示例文本", "另一个文本示例", ...]
# labels = [0, 1, ...]  # 假设0和1代表不同的类别


'''
Keras中的`Tokenizer`类在进行文本处理时，默认遵循以下几条规则：
1. **分词**：默认情况下，`Tokenizer`不会执行复杂的分词操作，而是将文本视为由空格分隔的tokens（词或符号）。
这意味着，如果你的文本数据包含未被空格分隔的词（例如，中文文本），则需要在调用`Tokenizer`之前先进行适当的分词处理（如使用jieba分词器）。
2. **过滤低频和高频词**：`Tokenizer`允许通过`min_df`和`max_df`参数来控制哪些单词会被纳入词汇表。
如果不显式设置，`min_df`默认为1，意味着所有在文档中至少出现一次的词都会被包括；`max_df`默认没有上限，
即所有词不论出现频率多高都会被包含。你可以通过设置这些参数来过滤低频或高频词。
3. **词汇表构建**：`Tokenizer`会自动构建一个基于训练数据的词汇表，其中每个唯一的token（词或符号）对应一个索引。
默认情况下，词汇表的大小是没有限制的，但可以通过`num_words`参数来限制词汇表中最多包含的单词数，
超出此数目的单词将被视为OOV（out-of-vocabulary）。
4. **OOV处理**：当文本中有词汇表之外的单词时，可以通过设置`oov_token`参数来指定一个特殊的标记，
代表所有OOV单词。默认情况下，如果遇到OOV单词，`Tokenizer`不会抛出错误或警告，而是简单忽略这些词。
5. **索引分配**：词汇表中的每个词被分配一个整数索引，索引从1开始（0通常保留为padding或OOV）。
高频词可能获得较低的索引，但具体索引分配还受词汇表构建时的具体算法影响，且通常对模型性能影响不大。
综上所述，`Tokenizer`默认行为主要是基于空格分隔文本，构建一个无大小限制的词汇表，并为每个词分配索引，
对于OOV处理没有默认标记，需要手动设置。根据具体任务需求，这些默认设置可能需要调整以优化模型性能。
'''
# 使用Tokenizer进行词汇化

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_content)
# tokenizer.fit_on_texts(validation_content)

train_sequences = tokenizer.texts_to_sequences(train_content)
validation_sequences=tokenizer.texts_to_sequences(validation_content)
test_sequences=tokenizer.texts_to_sequences(test_content)


'''
这段代码的作用是为了确保所有的序列（在这里特指经过分词并转换为整数序列的文本数据）具有相同的长度，
这对于大多数基于序列的数据处理任务（如使用循环神经网络RNN、LSTM或Transformer等模型）是必要的，
因为这些模型通常要求输入具有固定的尺寸。下面是代码含义的详细解释：

    maxlen = 100: 这行代码设定了所有序列将被调整到的统一长度。在这个例子中，最大长度被设定为100。
    这意味着如果某个文本转换成的序列长度少于100，将会被填充；如果超过100，则会进行截断。这个值应该根据实际数据分布和模型需求来设定。
    pad_sequences(sequences, maxlen=maxlen, padding='post'):
        sequences: 是之前通过 tokenizer.texts_to_sequences(texts) 得到的一系列整数序列列表，表示了文本数据。
        maxlen=100: 指定所有序列都将被调整到的长度。短于这个长度的序列会被填充，长于这个长度的序列会被截断。
        padding='post': 指定了填充的方式。'post' 表示在序列的末尾添加填充项。相反，如果设置为 'pre'，
        则会在序列的开头添加填充。填充的值通常是0或其他特殊标记，具体取决于实现和上下文。
        函数 pad_sequences 会返回一个新的序列列表，其中每个序列都被调整为长度100，不足的部分按照指定的填充策略补充。

'''
# 序列填充，保证所有序列长度一致
maxlen = 15  # 根据实际情况设定
x_train = pad_sequences(train_sequences, maxlen=maxlen, padding='post')
x_validation = pad_sequences(validation_sequences, maxlen=maxlen, padding='post')
x_test=pad_sequences(test_sequences, maxlen=maxlen, padding='post')

'''
这段代码是使用Keras库构建一个基于LSTM（长短时记忆网络）的深度学习模型，用于处理文本数据的二分类任务（例如，判断文本是正面情绪还是负面情绪）。下面是逐行的解释：
    embedding_dim = 128: 定义了词嵌入向量的维度，即每个词将被映射到一个128维的向量空间中，
    这个维度的选择是基于任务复杂性和计算资源的权衡。

    Sequential(): 创建一个线性的神经网络模型，模型将按顺序添加的层依次堆叠。

    model.add(Embedding(10000, embedding_dim, input_length=maxlen)): 添加一个词嵌入层。这里的参数意义分别为：
        10000 表示词汇表的大小，即模型能处理的唯一单词数量。
        embedding_dim=128 是嵌入向量的维度，每个词将被映射到这个维度的空间中。
        input_length=maxlen 指定了输入序列的长度，之前设定为100，保证所有输入文本序列长度一致。

    model.add(SpatialDropout1D(0.2)): 添加了空间dropout层，以1D形式应用在LSTM的输入上，丢弃率为0.2，用于防止过拟合。

    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2)): 添加了一个LSTM层，参数说明如下：
        64 表示LSTM单元的数量，即输出的维度。
        dropout=0.2 表示在LSTM的输入和输出上应用dropout，丢弃率为0.2。
        recurrent_dropout=0.2 表示在LSTM单元内部的循环连接上应用dropout，同样为0.2，进一步防止过拟合。

    model.add(Dense(1, activation='sigmoid')): 添加了一个全连接层（Dense层），
    输出单元数为1（因为是二分类问题），激活函数使用sigmoid，将输出压缩到(0, 1)区间，适合二分类概率输出。

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']): 编译模型，指定：
        损失函数为binary_crossentropy，这是二分类任务常用的损失函数，适合与sigmoid激活函数配合。
        优化器为adam，一种高效的梯度下降优化算法，自适应学习率。
        评价指标为accuracy，即准确率，用于评估模型性能。
'''

embedding_dim = 128  # 嵌入向量的维度

model = Sequential()
model.add(Embedding(10000, embedding_dim, input_length=maxlen))


from keras import regularizers
# 添加LSTM层，可以使用SpatialDropout1D在LSTM的输入或输出上
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2,
               recurrent_dropout=0.2,
               return_sequences=True,
               kernel_regularizer=regularizers.l2(),  # 添加L2正则化到权重
               bias_regularizer=regularizers.l2()
               ))
model.add(BatchNormalization())  # 在LSTM层之后添加BN层
# model.add(Dropout(rate=0.2))           # 在LSTM层之后添加Dropout层，丢弃率为20%

# model.add(Dense(1, activation='sigmoid'))  # 对于二分类问题，使用sigmoid激活函数

# 添加其他层，例如，如果有多层LSTM，或者最终的全连接层
'''
这串代码是用在Keras库中构建模型时的一行命令，用于添加一个LSTM（长短时记忆）层到神经网络模型中。下面是对该行代码各部分参数含义的详细解释：

- `model.add(...)`：这表示向现有的模型(`model`)中添加一个新的层。

- `LSTM(64,...)`：这部分指定了要添加的层是一个LSTM层，其中的数字`64`代表该LSTM层输出的维度，
也就是隐藏单元的数量。较大的数值可以学习更复杂的模式，但也会增加计算资源的需求和训练时间。

- `dropout=0.2`：这里的`dropout`参数是在LSTM层的输入部分应用的Dropout比率。
Dropout是一种正则化技术，通过在训练过程中随机“丢弃”（设置为0）一定比例的输入神经元来减少过拟合。在这个例子中，20%的输入神经元会被临时丢弃。

- `recurrent_dropout=0.2`：这是在LSTM层的循环（或称为“递归”）连接中应用的Dropout比率。
与常规的dropout不同，`recurrent_dropout`仅影响到时间序列中的隐藏状态传递，即在不同时间步之间，会有20%的隐藏单元被丢弃。这个参数可以帮助模型学习长期依赖，同时减轻过拟合。

综上所述，这行代码的作用是向模型中添加一个具有64个隐藏单元的LSTM层，
并在输入和循环连接中分别应用20%的Dropout策略，以提高模型的泛化能力。
'''
model.add(LSTM(64, dropout=0.2,
               recurrent_dropout=0.2,
               kernel_regularizer=regularizers.l2(),
               bias_regularizer=regularizers.l2()))
model.add(BatchNormalization())  # 在LSTM层之后添加BN层
model.add(Dropout(rate=0.2))  # 可以在每个LSTM之后都加，或者只在某些关键位置加



model.add(Dense(10,activation='softmax'))


from keras import optimizers
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.Nadam(),#optimizers.RMSprop
              metrics=['accuracy'])



from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 训练模型
epochs = 10
batch_size = 256
history = model.fit(
    x_train,
    train_label,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_validation, validation_label))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

from matplotlib import pyplot as plt

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

print(model.summary())
result=model.evaluate(x_test,test_label)

print(result)

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 数据生成
# 这里我们使用make_blobs生成模拟数据，它能创建多个高斯分布的数据点
n_samples = 300
random_state = 42
n_components = 3  # 假设数据由3个高斯分布混合而成

X, y = make_blobs(n_samples=n_samples, centers=n_components,
                  random_state=random_state, cluster_std=0.6)

# 使用GaussianMixture模型
gmm = GaussianMixture(n_components=n_components, random_state=random_state)
gmm.fit(X)

# 预测每个样本属于各个高斯分布的概率
probabilities = gmm.predict_proba(X)

# 可视化结果
plt.figure(figsize=(10, 5))

# 绘制数据点，颜色根据GMM分配的最可能类别
plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), s=40, cmap='viridis')

# 绘制GMM的高斯成分中心和边界
colors = ['r', 'g', 'b']
for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm.covariances_, colors)):
    v, w = np.linalg.eigh(covar)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1],
                              180 + angle, color=color,
                              linewidth=2, fill=False)
    plt.gca().add_artist(ell)

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title('Gaussian Mixture Model')
plt.xlabel('Feature space for X')
plt.ylabel('Feature space for Y')
plt.show()

'''