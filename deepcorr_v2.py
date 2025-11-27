import numpy as np
import pickle
import tqdm
import datetime
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

flow_size = 300
negative_samples = 199
num_epochs = 200
batch_size = 256
learn_rate = 0.0001
valid_freq = 3000

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


def generate_data(dataset, train_index, test_index, flow_size):
    all_samples = len(train_index)
    labels = np.zeros((all_samples * (negative_samples + 1), 1), dtype=np.float32)
    l2s = np.zeros((all_samples * (negative_samples + 1), 8, flow_size, 1), dtype=np.float32)

    index = 0
    random_ordering = [] + train_index
    for i in tqdm.tqdm(train_index, desc="Generating train data"):
        l2s[index, 0, :, 0] = np.array(dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0
        l2s[index, 1, :, 0] = np.array(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0
        l2s[index, 2, :, 0] = np.array(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0
        l2s[index, 3, :, 0] = np.array(dataset[i]['here'][0]['->'][:flow_size]) * 1000.0

        l2s[index, 4, :, 0] = np.array(dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0
        l2s[index, 5, :, 0] = np.array(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0
        l2s[index, 6, :, 0] = np.array(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0
        l2s[index, 7, :, 0] = np.array(dataset[i]['here'][1]['->'][:flow_size]) / 1000.0

        if index % (negative_samples + 1) != 0:
            print(index)
            raise ValueError("Index alignment error")

        labels[index, 0] = 1
        m = 0
        index += 1

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

    l2s_test = np.zeros((len(test_index) * (negative_samples + 1), 8, flow_size, 1), dtype=np.float32)
    labels_test = np.zeros((len(test_index) * (negative_samples + 1),), dtype=np.float32)
    index = 0
    random_test = [] + test_index

    for i in tqdm.tqdm(test_index, desc="Generating test data"):
        if index % (negative_samples + 1) != 0:
            print(index)
            raise ValueError("Index alignment error")

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


def build_keras_model(flow_size, dropout_rate=0.4):
    inp = tf.keras.Input(shape=(8, flow_size, 1), name='flow_input')

    x = tf.keras.layers.Conv2D(filters=2000, kernel_size=(2, 20), 
                               strides=(2, 2), padding='valid', 
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                               bias_initializer='zeros')(inp)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 5), strides=(1, 1), 
                                  padding='valid')(x)

    x = tf.keras.layers.Conv2D(filters=800, kernel_size=(4, 10), 
                               strides=(2, 2), padding='valid',
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                               bias_initializer='zeros')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 1), 
                                  padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(3000, 
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(800,
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(100,
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    logits = tf.keras.layers.Dense(1, activation=None, 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    bias_initializer='zeros',
                                    name='logits')(x)

    model = tf.keras.Model(inputs=inp, outputs=logits, name='DeepCorr_Keras')
    return model


class StepBasedValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, l2s_test, labels_test, epoch_threshold):
        super().__init__()
        self.l2s_test = l2s_test
        self.labels_test = labels_test
        self.epoch_threshold = epoch_threshold
        self.global_step = 0

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        
        if self.global_step % valid_freq == 0:
            current_epoch = self.global_step // (len(self.l2s_test) // batch_size + 1)
            if current_epoch < self.epoch_threshold:
                return
                
            print(f"\n[Step {self.global_step}] Running validation...")
            self._run_validation(current_epoch)

    def _run_validation(self, epoch):
        num_samples = len(self.l2s_test)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        all_preds = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch = self.l2s_test[start_idx:end_idx]
            
            preds = self.model.predict(batch, verbose=0, batch_size=len(batch))
            all_preds.append(preds.reshape(-1))
        
        preds = np.concatenate(all_preds)
        
        num_samples_test = len(self.l2s_test) // (negative_samples + 1)
        tp = 0
        fp = 0

        for idx in range(num_samples_test - 1):
            seg_start = idx * (negative_samples + 1)
            seg_end = (idx + 1) * (negative_samples + 1)
            seg = preds[seg_start:seg_end]
            best = np.argmax(seg)
            label_index = best + seg_start
            
            if self.labels_test[label_index] == 1:
                tp += 1
            else:
                fp += 1

        acc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        print(f"[Step {self.global_step}] Validation -> tp={tp}, fp={fp}, acc={acc:.4f}\n")

        if acc > 0.75: # original is 0.8
            ckpt_path = f'./ckpt/tordata{flow_size}_step{self.global_step}_acc{acc:.3f}.weights.h5'
            os.makedirs('./ckpt', exist_ok=True)
            self.model.save_weights(ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")


def train_mode():
    len_tr = len(dataset)
    train_ratio = float(len_tr - 6000) / float(len_tr)
    rr = list(range(len(dataset)))
    np.random.shuffle(rr)
    
    train_index = rr[:int(len_tr * train_ratio)]
    test_index = rr[int(len_tr * train_ratio):]
    
    os.makedirs('./ckpt', exist_ok=True)
    pickle.dump(test_index, open(f'./ckpt/tordata{flow_size}_test_index.pickle', 'wb'))

    print("Generating training and test data...")
    l2s, labels, l2s_test, labels_test = generate_data(
        dataset=dataset, 
        train_index=train_index, 
        test_index=test_index, 
        flow_size=flow_size
    )

    print(f"Train samples: {len(l2s)}, Test samples: {len(l2s_test)}")

    model = build_keras_model(flow_size, dropout_rate=0.4)
    
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    
    model.compile(optimizer=optimizer, loss=loss_fn)

    log_dir = f'./logs/allcir_{flow_size}_{datetime.datetime.now()}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    
    val_callback = StepBasedValidationCallback(
        l2s_test, labels_test, epoch_threshold=1
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        if epoch > 0:
            print("Regenerating training data with new negative samples...")
            l2s, labels, l2s_test, labels_test = generate_data(
                dataset=dataset, 
                train_index=train_index, 
                test_index=test_index, 
                flow_size=flow_size
            )
        
        rr = list(range(len(l2s)))
        np.random.shuffle(rr)
        l2s = l2s[rr]
        labels = labels[rr]
        
        ds = tf.data.Dataset.from_tensor_slices((l2s, labels))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        model.fit(
            ds, 
            epochs=1,
            callbacks=[val_callback, tensorboard_callback],
            verbose=1
        )

    final_path = f'./ckpt/tordata{flow_size}_final_weights.weights.h5'
    model.save_weights(final_path)
    print(f"Training complete. Final weights saved: {final_path}")


def test_mode():
    test_index = pickle.load(open(f'./ckpt/tordata{flow_size}_test_index.pickle', 'rb'))[:1000]
    
    batch_size = int(2804 / 2)
    model = build_keras_model(flow_size, dropout_rate=0.0)

    name = input('Model weights path: ').strip()
    model.load_weights(name)
    print(f"Model weights loaded from {name}")

    corrs = np.zeros((len(test_index), len(test_index)))
    l2s_test_all = np.zeros((batch_size, 8, flow_size, 1), dtype=np.float32)
    l_ids = []
    index = 0
    
    xi = 0
    for i in tqdm.tqdm(test_index, desc="Testing"):
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
                preds = model.predict(l2s_test_all, batch_size=batch_size, verbose=0)
                for ids in range(len(l_ids)):
                    di, dj = l_ids[ids]
                    corrs[di, dj] = preds[ids, 0]
                l_ids = []
            xj += 1
        xi += 1

    output_file = f'./ckpt/tordata{flow_size}_correlation_values_test.npy'
    np.save(output_file, corrs)
    print(f"Correlation matrix saved: {output_file}")


if __name__ == '__main__':
    mode = input("Train mode? (y/n): ").strip().lower()
    if mode == 'y':
        train_mode()
    else:
        test_mode()