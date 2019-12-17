import tensorflow as tf
import os

class SGF():
    def __init__(self):
        self.POS = 'abcdefghijklmno'

    def alphatoNum(self, path):
        f = open(path, 'r')
        data = f.read()
        f.close()
        data = data.split(";")
        board = []
        step = 0
        for point in data[2:-1]:
            if not self.POS.find(point[2]) and not self.POS.find(point[3]):
                x = self.POS.index(point[2])
                y = self.POS.index(point[3])
                color = step % 2 + 1
                step += 1
                board.append([x, y, color, step])
        return board

    def datatoTrain(self, path, color):
        data = self.alphatoNum(path)
        total_step = len(data)
        train_x = []
        train_y = []
        player = 1.0
        tmp = [0.0 for i in range(225)]
        for step in range(total_step):
            y = [0.0 for i in range(225)]
            train_x.append(tmp.copy())
            tmp[data[step][0] * 15 + data[step][1]] = player
            player = 2.0 if player == 1.0 else 1.0
            y[data[step][0] * 15 + data[step][1]] = 1.0
            train_y.append(y.copy())
        return train_x, train_y

    @staticmethod
    def getSgf(path):
        root = os.listdir(path)
        files = []
        for p in root:
            child = os.path.join("%s%s" % (path, p))
            files.append(child)
        return files

class CNN():
    def __init__(self):
        # initialize
        self.sess = tf.compat.v1.InteractiveSession()

        # paras
        self.W_conv1 = self.weight_varible([5, 5, 1, 32])
        self.b_conv1 = self.bias_variable([32])

        # conv layer-1
        self.x = tf.compat.v1.placeholder(tf.float32, [None, 225])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, 225])
        self.x_image = tf.compat.v1.reshape(self.x, [-1, 15, 15, 1])
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        #   conv layer-2
        self.W_conv2 = self.weight_varible([5, 5, 32, 64])
        self.b_conv2 = self.bias_variable([64])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        # full connection
        self.W_fc1 = self.weight_varible([4 * 4 * 64, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool1, [-1, 4 * 4 * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        # dropout
        # self.keep_prob = tf.placeholder(tf.float32)
        # self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # output layer: softmax
        self.W_fc2 = self.weight_varible([1024, 225])
        self.b_fc2 = self.bias_variable([225])

        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)

        # model training
        self.sorted_pred = tf.argsort(self.y_conv, direction="DESCENDING")
        self.cross_entropy = -tf.reduce_sum(self.y * tf.math.log(self.y_conv))
        self.train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.compat.v1.train.Saver()

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    @staticmethod
    def weight_varible(shape):
        initial = tf.random.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def expend(self, board):
        new = [[0.0 for i in range(15)] for i in range(15)]
        size = len(board)
        extra = (15 - size) / 2
        extra = int(extra)
        for i in range(size):
            for l in range(size):
                new[extra + i][extra + l] = board[i][l]
        return new, extra

    def prediction(self,board):
        new_board = board
        extra = 0
        if len(board) < 15:
           new_board,extra= self.expend(board)
        data = []
        tmp = []
        result = []
        finded = 0
        for row in new_board:
            for point in row:
                    tmp.append(point)
        data.append(tmp)
        left_col = extra
        right_col = extra+len(board)-1
        top_row = extra
        bottom_row = extra+len(board)-1
        sorted = self.sess.run(self.sorted_pred,feed_dict={self.x:data})
        for dis in sorted[0]:
            col = dis%15
            if dis < 15:
                row = 0
            else:
                row = (dis - col)/15
                row = int(row)
            if left_col <= col <= right_col and top_row <= row <= bottom_row:
                col = col - extra
                row = row-extra
                if board[row][col] == 0.0:
                    finded += 1
                    result.append([row,col])
            if finded >= 4:
                break
        return result

    def save(self, path):
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)

# save training result as model.ckpt
# if __name__ == "__main__":
#     _cnn = CNN()
#     sgf = SGF()
#     batch = 0
#     files = sgf.getSgf('.\SGF\\')
#     trainFile = files[:1000]
#     for f in trainFile:
#         x, y = sgf.datatoTrain(f,1)
#         _cnn.sess.run(_cnn.train_step, feed_dict={_cnn.x: x, _cnn.y: y})
#         batch += 1
#         print(batch)
#     _cnn.save('.\model\model.ckpt')
