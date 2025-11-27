import numpy as np
import tqdm
import pickle
import tensorflow as tf
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

tf.compat.v1.disable_eager_execution()

flow_size = 300
negative_samples = 199
num_epochs = 200

is_training = input('train? (y/n)')
TRAINING = True if is_training == 'y' else False

all_runs = {
    '8872':'192.168.122.117','8802':'192.168.122.117',
    '8873':'192.168.122.67','8803':'192.168.122.67',
    '8874':'192.168.122.113','8804':'192.168.122.113',
    '8875':'192.168.122.120','8876':'192.168.122.30',
    '8877':'192.168.122.208','8878':'192.168.122.58'
}

dataset = []
for name in all_runs:
    dataset += pickle.load(open(f'./dataset/{name}_tordata{flow_size}.pickle', 'rb'))

if TRAINING:
    len_tr = len(dataset)
    train_ratio = float(len_tr - 6000) / float(len_tr)
    rr = list(range(len(dataset)))
    np.random.shuffle(rr)

    train_index = rr[:int(len_tr * train_ratio)]
    test_index = rr[int(len_tr * train_ratio):]
    pickle.dump(test_index, open(f'test_index{flow_size}.pickle', 'wb'))

else:
    test_index = pickle.load(open(f'test_index{flow_size}.pickle', 'rb'))[:1000]


def generate_data(dataset, train_index, test_index, flow_size):
    global negative_samples

    all_samples = len(train_index)
    labels = np.zeros((all_samples * (negative_samples + 1), 1), dtype = np.float32)
    l2s = np.zeros((all_samples * (negative_samples + 1), 8, flow_size, 1), dtype = np.float32)

    index = 0
    random_ordering = [] + train_index

    for i in tqdm.tqdm(train_index):
        l2s[index, 0, :, 0] = np.array(dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0
        l2s[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
        l2s[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
        l2s[index, 3, :, 0] = np.array(dataset[i]['here'][0]['->'][:flow_size]) * 1000.0

        l2s[index, 4, :, 0] = np.array(dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0
        l2s[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
        l2s[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
        l2s[index, 7, :, 0] = np.array(dataset[i]['here'][1]['->'][:flow_size]) / 1000.0


        if index % (negative_samples + 1) != 0:
            print(index) # , len(nears))
            raise
        
        labels[index, 0] = 1
        
        index += 1

        m = 0
        np.random.shuffle(random_ordering)
        for idx in random_ordering:
            if idx == i or m > (negative_samples - 1):
                continue
            m += 1
            
            l2s[index, 0, :, 0] = np.array(dataset[idx]['here'][0]['<-'][:flow_size]) * 1000.0
            l2s[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
            l2s[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
            l2s[index, 3, :, 0] = np.array(dataset[idx]['here'][0]['->'][:flow_size]) * 1000.0

            l2s[index, 4, :, 0] = np.array(dataset[idx]['here'][1]['<-'][:flow_size]) / 1000.0
            l2s[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
            l2s[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
            l2s[index, 7, :, 0] = np.array(dataset[idx]['here'][1]['->'][:flow_size]) / 1000.0

            labels[index, 0] = 0
            index += 1

    l2s_test = np.zeros((len(test_index) * (negative_samples + 1), 8, flow_size, 1), dtype = np.float32)
    labels_test = np.zeros((len(test_index) * (negative_samples + 1)), dtype = np.float32)
    index = 0
    random_test = [] + test_index

    for i in tqdm.tqdm(test_index):
        if index % (negative_samples + 1) != 0:
            print(index) #, nears
            raise
        m = 0

        np.random.shuffle(random_test)
        for idx in random_test:
            if idx == i or m > (negative_samples - 1):
                continue
            m += 1

            l2s_test[index, 0, :, 0] = np.array(dataset[idx]['here'][0]['<-'][:flow_size]) * 1000.0
            l2s_test[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
            l2s_test[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
            l2s_test[index, 3, :, 0] = np.array(dataset[idx]['here'][0]['->'][:flow_size]) * 1000.0

            l2s_test[index, 4, :, 0] = np.array(dataset[idx]['here'][1]['<-'][:flow_size]) / 1000.0
            l2s_test[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
            l2s_test[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
            l2s_test[index, 7, :, 0] = np.array(dataset[idx]['here'][1]['->'][:flow_size]) / 1000.0
            labels_test[index] = 0
            index += 1

        l2s_test[index, 0, :, 0] = np.array(dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0
        l2s_test[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
        l2s_test[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
        l2s_test[index, 3, :, 0] = np.array(dataset[i]['here'][0]['->'][:flow_size]) * 1000.0

        l2s_test[index, 4, :, 0] = np.array(dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0
        l2s_test[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
        l2s_test[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
        l2s_test[index, 7, :, 0] = np.array(dataset[i]['here'][1]['->'][:flow_size]) / 1000.0

        labels_test[index] = 1
        index += 1

    return l2s, labels, l2s_test, labels_test


def model(flow_before, dropout_keep_prob):
    last_layer = flow_before
    flat_layers_after = [flow_size * 2, 1000, 50, 1]
    for l in range(len(flat_layers_after) - 1):
        flat_weight = tf.compat.v1.get_variable("flat_after_weight%d" %l, [flat_layers_after[l], flat_layers_after[l + 1]], 
                                      initializer = tf.random_normal_initializer(stddev = 0.01, mean = 0.0))

        flat_bias = tf.compat.v1.get_variable("flat_after_bias%d" %l, [flat_layers_after[l + 1]], 
                                    initializer = tf.zeros_initializer())

        _x = tf.add(tf.matmul(last_layer, flat_weight), flat_bias)
        if l < len(flat_layers_after) - 2:
            _x = tf.nn.dropout(tf.nn.relu(_x, name = 'relu_noise_flat_%d' %l), rate = 1.0 - dropout_keep_prob)
        last_layer = _x
    return last_layer
        

def model_cnn(flow_before, dropout_keep_prob):
    last_layer = flow_before

    CNN_LAYERS = [[2, 20, 1, 2000, 5], [4, 10, 2000, 800, 3]]
    
    for cnn_size in range(len(CNN_LAYERS)):
        cnn_weights = tf.compat.v1.get_variable("cnn_weight%d" %cnn_size, 
                                      CNN_LAYERS[cnn_size][:-1], 
                                      initializer = tf.random_normal_initializer(stddev = 0.01))
        cnn_bias = tf.compat.v1.get_variable("cnn_bias%d" %cnn_size, 
                                   [CNN_LAYERS[cnn_size][-2]], 
                                   initializer=tf.zeros_initializer())

        _x = tf.nn.conv2d(last_layer, cnn_weights, strides = [1, 2, 2, 1], padding = 'VALID')
        _x = tf.nn.bias_add(_x, cnn_bias)
        conv = tf.nn.relu(_x, name = 'relu_cnn_%d' %cnn_size)
        pool = tf.nn.max_pool(conv, ksize = [1, 1, CNN_LAYERS[cnn_size][-1], 1], 
                              strides = [1, 1, 1, 1], 
                              padding = 'VALID')
        last_layer = pool

    last_layer = tf.reshape(last_layer, [batch_size, -1])
    
    flat_layers_after = [49600, 3000, 800, 100, 1]
    for l in range(len(flat_layers_after) - 1):
        flat_weight = tf.compat.v1.get_variable("flat_after_weight%d" %l, 
                                      [flat_layers_after[l], 
                                       flat_layers_after[l + 1]], 
                                       initializer = tf.random_normal_initializer(stddev = 0.01, mean = 0.0))

        flat_bias = tf.compat.v1.get_variable("flat_after_bias%d" %l, 
                                    [flat_layers_after[l + 1]], 
                                    initializer = tf.zeros_initializer())

        _x = tf.add(tf.matmul(last_layer, flat_weight), flat_bias)
        if l < len(flat_layers_after) - 2:
            _x = tf.nn.dropout(tf.nn.relu(_x, name = 'relu_noise_flat_%d' %l), rate = 1.0 - dropout_keep_prob)
        last_layer = _x
    return last_layer


if TRAINING:
    batch_size = 256
    learn_rate = 0.0001

    graph = tf.Graph()
    with graph.as_default():
        train_flow_before = tf.compat.v1.placeholder(tf.float32, shape = [batch_size, 8, flow_size, 1], name = 'flow_before_placeholder')
        train_label = tf.compat.v1.placeholder(tf.float32, name = 'label_placeholder', shape = [batch_size, 1])
        dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name = 'dropout_placeholder')

        y2 = model_cnn(train_flow_before, dropout_keep_prob)
        predict = tf.nn.sigmoid(y2)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y2, labels = train_label), name = 'loss_sigmoid')

        optimizer = tf.compat.v1.train.AdamOptimizer(learn_rate).minimize(loss)
        s_loss = tf.compat.v1.summary.scalar('loss', loss)
        summary_op = tf.compat.v1.summary.merge_all()

        init = tf.compat.v1.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()
else:
    batch_size = int(2804 / 2)

    graph = tf.Graph()
    with graph.as_default():
        train_flow_before = tf.compat.v1.placeholder(tf.float32, shape = [batch_size, 8, flow_size, 1], name = 'flow_before_placeholder')
        train_label = tf.compat.v1.placeholder(tf.float32, name = 'label_placeholder',shape = [batch_size, 1])
        dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name = 'dropout_placeholder')

        y2 = model_cnn(train_flow_before, dropout_keep_prob)
        predict = tf.nn.sigmoid(y2)
        saver = tf.compat.v1.train.Saver()

writer = tf.compat.v1.summary.FileWriter(f'./logs/allcir_{flow_size}_' + str(datetime.datetime.now()), graph = graph)

if TRAINING:
    with tf.compat.v1.Session(graph = graph) as session:
        session.run(init)
        
        for epoch in range(num_epochs):
            l2s, labels, l2s_test, labels_test = generate_data(dataset = dataset, train_index = train_index, test_index = test_index, flow_size = flow_size)
            rr = list(range(len(l2s)))
            np.random.shuffle(rr)
            l2s = l2s[rr]
            labels = labels[rr]

            average_loss = 0
            new_epoch = True
            num_steps = (len(l2s) // batch_size) - 1

            for step in range(num_steps):
                start_ind = step * batch_size
                end_ind = ((step + 1) * batch_size)
                if end_ind < start_ind:
                    print('HOOY')
                    continue
                else:
                    batch_flow_before = l2s[start_ind:end_ind, :]
                    batch_label = labels[start_ind:end_ind]

                feed_dict = {train_flow_before: batch_flow_before,
                             train_label: batch_label,
                             dropout_keep_prob: 0.6}
                _, loss_val, summary = session.run([optimizer, loss, summary_op], feed_dict = feed_dict)

                writer.add_summary(summary, (epoch * num_steps) + step)

                if ((epoch * num_steps) + step) % 100 == 0:
                    print("Average loss on validation set at step ", (epoch * num_steps) + step, ": ", loss_val)
                if (((epoch * num_steps) + step)) % 3000 == 0 and epoch > 1:
                    tp = 0
                    fp = 0

                    num_steps_test = (len(l2s_test) // batch_size) - 1
                    Y_est = np.zeros((batch_size * (num_steps_test + 1)))
                    
                    for step in range(num_steps_test):
                        start_ind = step * batch_size
                        end_ind = ((step + 1) * batch_size)
                        test_batch_flow_before = l2s_test[start_ind:end_ind]
                        feed_dict = {train_flow_before: test_batch_flow_before, 
                                     dropout_keep_prob: 1.0}

                        est = session.run(predict, feed_dict = feed_dict)
                        Y_est[start_ind:end_ind] = est.reshape((batch_size))
                    
                    num_samples_test = len(l2s_test) // (negative_samples + 1)

                    for idx in range(num_samples_test - 1):
                        best = np.argmax(Y_est[idx * (negative_samples + 1):(idx + 1) * (negative_samples + 1)])

                        if labels_test[best + (idx * (negative_samples + 1))] == 1:
                            tp += 1
                        else:
                            fp += 1

                    print(tp, fp)
                    acc = float(tp) // float(tp + fp)
                    if float(tp) // float(tp + fp) > 0.8:
                        print('saving...')
                        save_path = saver.save(session, "./ckpt/tor_199_epoch%d_step%d_acc%.2f.ckpt" %(epoch, step, acc))
                        print('saved')
            print('Epoch', epoch)
else:
    with tf.compat.v1.Session(graph = graph) as session:
        name = input('model name: ')
        saver.restore(session, "./ckpt/%s" %name)
        print("Model restored.")
        corrs = np.zeros((len(test_index), len(test_index)))
        batch = []
        l2s_test_all = np.zeros((batch_size, 8, flow_size, 1), dtype = np.float32)
        l_ids = []
        index = 0
        xi, xj = 0, 0
        for i in tqdm.tqdm(test_index):
            xj = 0
            for j in test_index:
                l2s_test_all[index, 0, :, 0] = np.array(dataset[j]['here'][0]['<-'][:flow_size]) * 1000.0
                l2s_test_all[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
                l2s_test_all[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
                l2s_test_all[index, 3, :, 0] = np.array(dataset[j]['here'][0]['->'][:flow_size]) * 1000.0

                l2s_test_all[index, 4, :, 0] = np.array(dataset[j]['here'][1]['<-'][:flow_size]) / 1000.0
                l2s_test_all[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
                l2s_test_all[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
                l2s_test_all[index, 7, :, 0] = np.array(dataset[j]['here'][1]['->'][:flow_size]) / 1000.0
                l_ids.append((xi, xj))
                index += 1
                if index == batch_size:
                    index = 0
                    cor_vals = session.run(predict, 
                                           feed_dict = {train_flow_before: l2s_test_all, 
                                                        dropout_keep_prob: 1.0})
                    for ids in range(len(l_ids)):
                        di, dj = l_ids[ids]
                        corrs[di, dj] = cor_vals[ids]
                    l_ids = []
                xj += 1
            xi += 1

        np.save('correlation_values_test.npy', corrs)
        print('Saved correlation_values_test.npy')