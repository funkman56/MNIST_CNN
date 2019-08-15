'''
MNIST CNN
'''

import tensorflow as tf 
from mnist import read_data_sets
tf.reset_default_graph()

mnist = read_data_sets("./MNIST_data", one_hot = True)

# 參數
learning_rate = 0.001
epochs = 1000
batch_size = 100
D = 0.5                 # dropout %



def weight_variable(shape) :
    initial_value = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial_value)

def bias_variable(shape) :
    initial_value = tf.constant(0.1, shape = shape)
    return tf.Variable(initial_value)

def conv2d(x, W) :
    # x 輸入照片 , W 是 Kernal 或 可稱 Filter
    # 步長 stride = [1, 水平步長, 垂直步長, 1] "通常前後都寫1"
    # padding = SAME mean add padding on input data 
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding= "SAME")

def max_pool_2x2(x) :
    # ksize = [1, 寬, 高, 1] "通常前後都寫1"
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides =[1, 2, 2, 1], padding="SAME")

W_conv1 = weight_variable([5, 5, 1, 32]) # 32張 5x5x1 的 kernal
b_conv1 = bias_variable([32])

x = tf.placeholder(tf.float32 , [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Reshape 的原因是 因為要轉成 CNN 可讀取的檔案格式
# Reshape our data as 4D tensorflow ,  batch of image 是 4D tensor
x_image = tf.reshape(x, [-1, 28, 28, 1]) # -1 電腦自動補資料
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 32 不能動 因為前一個feature map 是 32張， 5 5 64 都可更改
W_conv2 = weight_variable([5, 5, 32, 64])  # 64 張 5x5x32 的 kernal
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

print(h_pool2)                             # 因為 tensor(shape=(?,7, 7, 64))  
W_fc1 = weight_variable([7*7*64 , 1024])   # 所以 7 * 7 * 64
b_fc1 = bias_variable([1024])

# flatten 之前的層，再把它們放進 fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout 增加準確率
# keep_prob 丟棄多少比例的neuron
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 定義 Cost 函數
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y , logits = y_conv))


# 訓練及選擇優化器
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)  
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy) 

# 計算準度 
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(epochs) :
        batch_xs,  batch_ys = mnist.train.next_batch(batch_size)

        train_step_ = sess.run(train_step, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: D})
        
        # print entropy every 50 steps
        if step % 50 == 0 :
            
            train_accuracy = sess.run(accuracy, feed_dict ={x: batch_xs, y: batch_ys, keep_prob: D})
            print("Step {}  : Training Accuracy is {}".format(step , train_accuracy ))

    # 讀取測試資料，驗證這個訓練模型
    # 測試時必須要存在所有的neuron 所以 keep_prob = 1.0
    test_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}) 
    print("\nTesting Accuracy =====> {}".format(test_accuracy))    