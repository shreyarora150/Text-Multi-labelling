import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy

# use the tokenizer from DistilRoBERTa

#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = transformers.TFRobertaModel.from_pretrained('roberta-base')
tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")


def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples, truncation=True, max_length=64,
                     padding="max_length")


def to_bow(example):
    """Converts the sequence of 1-hot vectors into a single many-hot vector"""
    vector = numpy.zeros(shape=(tokenizer.vocab_size,))
    vector[example["input_ids"]] = 1
    return {"input_bow": vector}


def train(model_path="model", train_path="train.csv", dev_path="dev.csv"):

    # load the CSVs into Huggingface datasets to allow use of the tokenizer
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # the labels are the names of all columns except the first
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        return {"labels": [float(example[l]) for l in labels]}

    # convert text and labels to format expected by model
    hf_dataset = hf_dataset.map(gather_labels)
    #hf_dataset = hf_dataset.map(tokenize, batched=True)
    #hf_dataset = hf_dataset.map(to_bow)

    # convert Huggingface datasets to Tensorflow datasets
    #train_dataset = hf_dataset["train"].to_tf_dataset(
    #    label_cols="labels",
    #    batch_size=16,
    #    shuffle=True)
    #dev_dataset = hf_dataset["validation"].to_tf_dataset(
    #    label_cols="labels",
    #    batch_size=16)

    train_dataset = pandas.DataFrame(hf_dataset["train"])
    dev_dataset = pandas.DataFrame(hf_dataset["validation"])

    X_train = train_dataset['text']
    text_list_train = [str(i) for i in X_train.values] 
    y_train = train_dataset.drop('text', axis=1).values
    X_dev = dev_dataset['text'].value_counts
    y_dev = dev_dataset.drop('text', axis=1).value_counts

    print(numpy.shape(X_train))
    print(numpy.shape(y_train))
    print(numpy.shape(X_dev))
    print(numpy.shape(y_dev))

    print(type(X_train))
    print(type(y_train))
    print(type(X_dev))
    print(type(y_dev))

    tokenized_inputs = tokenizer(text_list_train, padding='max_length', truncation=True, max_length = 64,return_tensors="tf")
    roberta_outputs = roberta_model(**tokenized_inputs)
    roberta_embeddings = roberta_outputs.last_hidden_state

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=64, input_shape=(None, roberta_embeddings.shape[-1])))
    model.add(tf.keras.laeyrs.Dense(units=32, activation='relu'))
    model.add(tf.keras.layersDense(units=7, activation='sigmoid'))


    #model.layers[0].set_weights([roberta_embeddings.numpy()])
    model.layers[0].trainable = False
    # define a model with a single fully connected layer
    """model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = 512, input_dim=tokenizer.vocab_size))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(len(labels), activation='sigmoid'))"""

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_f1_score',
    baseline=0.79,
    min_delta=0.001,
    patience=3,
    restore_best_weights=True)

    # specify compilation hyperparameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.002),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)])
    
    model_callback_params = tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_f1_score",
                mode="max",
                save_best_only=True)

    # fit the model to the training data, monitoring F1 on the dev data
    model.fit(train_dataset,
            epochs=10,
            validation_data=dev_dataset,
            callbacks=[early_stopping,model_callback_params])


def predict(model_path="model", input_path="dev.csv"):

    # load the saved model
    model = tf.keras.models.load_model(model_path)

    # load the data for prediction
    # use Pandas here to make assigning labels easier later
    df = pandas.read_csv(input_path)

    # create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset = hf_dataset.map(to_bow)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_bow",
        batch_size=16)

    # generate predictions from model
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()
