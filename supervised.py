import tensorflow as tf
import tensorflow_datasets
from matplotlib.pyplot import ylabel, plot, legend, show, xlabel

imdb, info = tensorflow_datasets.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder

sample_string = 'TensorFlow, from basics to mastery'

tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is: {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('The original string: {}'.format(original_string))

for ts in tokenized_string:
    print('{} ---> {}'.format(ts, tokenizer.decode([ts])))

embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

num_epochs = 10

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_data,
                    epochs=num_epochs,
                    validation_data=test_data)


def plot_graphs(history, string):
    plot(history.history[string])
    plot(history.history['val_' + string])
    xlabel("Epochs")
    ylabel(string)
    legend([string, 'val_' + string])
    show()


plot_graphs(history, "acc")
plot_graphs(history, "loss")
