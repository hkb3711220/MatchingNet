import tensorflow as tf

rnn = tf.nn.rnn_cell
slim = tf.contrib.slim

class MatchingNet(object):

    def __init__(self, N_way, k_shot, usefce=True, batch_size=5):

        #an N-way k-shot learning task
        #Each Method is providing with a set of k labelled examples
        #from each of N class

        #The task is then to classifty a disjoint batch of
        #unlabelled examples into one of thess N classed.

        self.img_size = 28
        self.batch_size = batch_size
        self.usefce = usefce
        self.N_way = N_way
        self.k_shot = k_shot

        self.processing_steps = 10
        self.support_set = tf.placeholder(tf.float32, [None, self.N_way*self.k_shot, self.img_size, self.img_size, 1])
        self.support_set_label = tf.placeholder(tf.int32, [None, self.N_way*self.k_shot, ])
        self.example_set = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        self.example_set_label = tf.placeholder(tf.int32, [None, ])

    def encode(self, inputs):

        #We used a simple yet powerful CNN as the embedding function – consisting of a stack of modules,
        #each of which is a 3 × 3 convolution with 64 filters followed by batch normalization [10], a Relu
        #non-linearity and 2 × 2 max-pooling. We resized all the images to 28 × 28 so that, when we stack 4
        #modules, the resulting feature map is 1 × 1 × 64, resulting in our embedding function f(x).

        with slim.arg_scope([slim.conv2d], num_outputs=64, kernel_size=3, normalizer_fn=slim.batch_norm):
            net = slim.conv2d(inputs)
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net)
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net)
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net)
            net = slim.max_pool2d(net, [2, 2])

        return tf.reshape(net, [-1, 1*1*64])

    def fce_g(self, support_set_encode):

        #support set S, g(xi, S), as a bidirectional LSTM. More precisely,
        #let g(xi) be a neural network (similar to fabove, e.g. aVGG or Inception model).
        #Then we define g(xi, S) = fw(hi) + bk(hi) + g(xi)

        fw_cell = rnn.LSTMCell(32)
        bk_cell = rnn.LSTMCell(32)

        #takes input and builds independent forward and backward RNNs with
        #the final forward and backward outputs depth-concatenated

        outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(fw_cell, bk_cell, support_set_encode, dtype=tf.float32)
        #such that the output will have the format [time][batch][cell_fw.output_size + cell_bw.output_size].
        return tf.add(tf.stack(support_set_encode), tf.stack(outputs))

    def fce_f(self, example_set_encode, gc_embedding):

        cell = rnn.LSTMCell(64)
        presv_state = cell.zero_state(self.batch_size, tf.float32)

        for i in range(self.processing_steps):
            #Since K Steps of 'reads'. The read-out hk is calculate
            output, state = cell(example_set_encode, presv_state) #(batch_size, 64) # state[0] is c, state[1] is h

            h_k = tf.add(output, example_set_encode)
            attention = tf.nn.softmax(tf.multiply(presv_state[1],gc_embedding))
            r_k = tf.reduce_sum(tf.multiply(attention, gc_embedding), axis=0)

            presv_state = rnn.LSTMStateTuple(state[0], tf.add(h_k, r_k))

        return output

    def cal_similarity(self, fc_embedding, gc_embedding):

        similarity_list = []
        #fc_embedding (batch_size, 64)
        #tf.unstack(gc_embedding, axis=0) (batch_size, 64)

        fc_embed_norm = tf.nn.l2_normalize(fc_embedding, axis=1)

        for i in tf.unstack(gc_embedding, axis=0):
            gc_embed_norm = tf.nn.l2_normalize(i, axis=1) #(batch_size, 64)
            #Inserts a dimension of 1 into a tensor's shape. (deprecated arguments)
            similarity = tf.matmul(tf.expand_dims(fc_embed_norm, 1), tf.expand_dims(gc_embed_norm, 2))
            similarity_list.append(similarity)

        return tf.squeeze(tf.stack(similarity_list, 1))

    def build(self):

        support_set_encode = [self.encode(i) for i in tf.unstack(self.support_set, axis=1)] #(batch_size, 64)
        gc_embedding = self.fce_g(support_set_encode) #(n*k, batch_size, 64)

        example_set_encode = self.encode(self.example_set) #(batch_size, 64)

        if self.usefce:
            fc_embedding = self.fce_f(example_set_encode, gc_embedding) #(batch_size, 64)
        else:
            fc_embedding = example_set_encode

        similarity = self.cal_similarity(fc_embedding, gc_embedding)
        attention = tf.nn.softmax(similarity)
        logits = tf.matmul(tf.expand_dims(attention, 1), tf.one_hot(self.support_set_label, self.N_way))
        predict = tf.argmax(logits, 1)
        print(logits)

        return tf.squeeze(logits, 1), predict #(batch, N_class * k_shot)

    def train(self, logits, label):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits())

MatchingNet(N_way=10, k_shot=1).build()
