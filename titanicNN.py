import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('train.csv')
X_columns = data.columns.tolist()
data2 = pd.read_csv('test.csv')

features = ["Pclass", "Sex", "SibSp", "Parch","Age"]


avg_values = data["Age"].mean()  # Izračunavanje prosečnih vrednosti za svaku kolonu
data["Age"] =data["Age"].fillna(avg_values)

avg_values = data2["Age"].mean()  # Izračunavanje prosečnih vrednosti za svaku kolonu
data2["Age"] =data2["Age"].fillna(avg_values)

y_train = data['Survived']
x_train = pd.get_dummies(data[features])
y_test = data2['Survived']
x_test = pd.get_dummies(data2[features])
x_train = np.reshape(x_train.values, (x_train.shape[0], -1))
x_test = np.reshape(x_test.values, (x_test.shape[0], -1))

nb_train = len(y_train)
nb_test = len(y_test)

# Parametri mreze
learning_rate = 0.001
nb_epochs = 16
batch_size = 9

# Parametri arhitekture
nb_input = 6  # MNIST data input (img shape: 28*28)
nb_hidden1 = 8  # 1st layer number of neurons
nb_hidden2 = 8  # 2nd layer number of neurons
nb_classes = 2   # MNIST total classes (0-1 digits)

# Sama mreza
w = {
    '1': tf.Variable(tf.random.normal([nb_input, nb_hidden1], dtype=tf.float64)),
    '2': tf.Variable(tf.random.normal([nb_hidden1, nb_hidden2], dtype=tf.float64)),
    'out': tf.Variable(tf.random.normal([nb_hidden2, nb_classes], dtype=tf.float64))
}

b = {
    '1': tf.Variable(tf.random.normal([nb_hidden1], dtype=tf.float64)),
    '2': tf.Variable(tf.random.normal([nb_hidden2], dtype=tf.float64)),
    'out': tf.Variable(tf.random.normal([nb_classes], dtype=tf.float64))
}

# w = {
#     '1': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_input, nb_hidden1], dtype=tf.float64)),
#     '2': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_hidden1, nb_hidden2], dtype=tf.float64)),
#     'out': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_hidden2, nb_classes], dtype=tf.float64))
# }

# b = {
#     '1': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_hidden1], dtype=tf.float64)),
#     '2': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_hidden2], dtype=tf.float64)),
#     'out': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_classes], dtype=tf.float64))
# }

f = {
    '1': tf.nn.relu,
    '2': tf.nn.relu,
    'out': tf.nn.softmax
}

def runNN(x):
    z1 = tf.add(tf.matmul(x, w['1']), b['1'])
    a1 = f['1'](z1)
    z2 = tf.add(tf.matmul(a1, w['2']), b['2'])
    a2 = f['2'](z2)
    z_out = tf.add(tf.matmul(a2, w['out']), b['out']) # a2 ovde!
    out = f['out'](z_out)

    pred = tf.argmax(out, 1)

    return pred, z_out

# GD je djubre
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
print(x_train)
# Trening!
for epoch in range(nb_epochs):
    epoch_loss = 0
    nb_batches = int(nb_train / batch_size)
    for i in range(nb_batches):
        x = x_train[i*batch_size : (i+1)*batch_size, :]
        y = y_train[i*batch_size : (i+1)*batch_size]
        y_onehot = tf.one_hot(y, nb_classes)

        with tf.GradientTape() as tape:
            _, z_out = runNN(x)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z_out, labels=y_onehot))

        w1_g, w2_g, wout_g, b1_g, b2_g, bout_g = tape.gradient(loss, [w['1'], w['2'], w['out'], b['1'], b['2'], b['out']])

        opt.apply_gradients(zip([w1_g, w2_g, wout_g, b1_g, b2_g, bout_g], [w['1'], w['2'], w['out'], b['1'], b['2'], b['out']]))

        epoch_loss += loss

    # U svakoj epohi ispisujemo prosečan loss.
    epoch_loss /= nb_train
    print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

# Test!
x = x_test
y = y_test

pred, _ = runNN(x)
pred_correct = tf.equal(pred, y)
accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))

print(f'Test acc: {accuracy:.3f}')