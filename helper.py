from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import re
import pickle
import ast
from keras.models import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import classification_report   
from sklearn.feature_extraction.text import TfidfVectorizer

class Helper(object):

    def dataset(self, file):
        # Membaca file dataset
        df = pd.read_csv(file)
        X = df['text'].tolist()
        y = df['label'].tolist()
        X_preprocessed = [self.preprocessing(data) for data in X]
        y = df["label"].map({"Spam": 1, "Ham": 0})
        df["preprocessing-result"] = X_preprocessed
        pd.DataFrame(df).to_csv('hasil_prapengolahan.csv', index=False, header=True)
        return X_preprocessed, y

    def preprocessing(self, text):
        # Cleaning
        text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+",
                      "", text)  # menghapus http/https (link)
        text = re.sub(r'[^\w\s]', ' ', text)  # menghilangkan tanda baca
        # mengganti karakter html dengan tanda petik
        text = re.sub('<.*?>', ' ', text)
        text = re.sub('[\s]+', ' ', text)  # menghapus spasi berlebihan
        # mempertimbangkan huruf dan angka
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub("\n", " ", text)  # mengganti line baru dengan spasi
        text = ' '.join(text.split())  # memisahkan dan menggabungkan kata

        # Case Folding
        text = text.lower()  # mengubah ke huruf kecil

        # Tokenize
        regexp = RegexpTokenizer(r'\w+|$[0-9]+|\S+')
        text = regexp.tokenize(text)

        # Stopword
        list_stopword = set(stopwords.words('indonesian'))
        hapus_kata = {"tidak", "enggak"}
        list_stopword.difference_update(hapus_kata)
        text = [token for token in text if token not in list_stopword]

        # Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        text = [stemmer.stem(word) for word in text]

        return text

    def split_data(self, X_preprocessed, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def tfidf_vectorizer(self, X_train, X_test):
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf
    
    def sequencepred(self, text):
        dataset = pd.read_csv("hasil_prapengolahan.csv")
        dataset["preprocessing-result"] = dataset["preprocessing-result"].apply(lambda x: ast.literal_eval(x))
        X_preprocessed = dataset["preprocessing-result"].tolist()
        tokenizer = Tokenizer()

        # loading
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        sequences = tokenizer.texts_to_sequences(X_preprocessed)

        max_length = max([len(s) for s in sequences])
        sequence = tokenizer.texts_to_sequences(text)
        padding = pad_sequences(sequence, maxlen=max_length)
        return padding
    
    def model_predict(self,input):
        model = load_model("lstm_spam_model.model")
        predict = model.predict(input)
        print(predict)
        labels = ["SPAM", "HAM"]
        if(predict[0] > 0.5):
            return labels[0]
        else:
            return labels[1]

    def change_input(self,text):
        text = [' '.join(text)]
        return text

    def sequence(self, X_preprocessed, X_train, X_test):
        max_words = 1000
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_preprocessed)

        # saving
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        sequences_train = tokenizer.texts_to_sequences(X_train)
        sequences_test = tokenizer.texts_to_sequences(X_test)

        length_sequences_train = max([len(s) for s in sequences_train])
        length_sequences_test = max([len(s) for s in sequences_test])
        if(length_sequences_train > length_sequences_test):
            max_length = length_sequences_train
        else :
            max_length = length_sequences_test
        print(max_length)

        X_train = pad_sequences(sequences_train, maxlen=max_length)
        X_test = pad_sequences(sequences_test, maxlen=max_length)
        return X_train, X_test, max_length
    

    def train_lstm_model(self, X_train, y_train, max_length, embedding_dim=128, epochs=10, batch_size=8, model_save_path='lstm_spam_model.model'):
        model = Sequential()
        model.add(Embedding(input_dim=1000,
                        output_dim=embedding_dim, input_length=max_length))
        model.add(LSTM(64, dropout=0.5))
        model.add(Dense(1, activation='sigmoid'))                                                     
        model.compile(loss='binary_crossentropy',
                    optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])

        print(model.summary())

        # Melatih model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

        #akurasi
        plt.plot(history.history['accuracy'], 'b',label='Training Accuracy')
        plt.plot(history.history['val_accuracy'],'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        #save
        plt.savefig('accuracy.png')
        plt.close()

        # loss
        plt.plot(history.history['loss'], 'b', label='Training Loss')
        plt.plot(history.history['val_loss'], 'r',label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        #save
        plt.savefig('loss.png')
        plt.close()
        
        tf.keras.models.save_model(model, model_save_path)

        return model

    def print_kinerja(self, model, X_test, y_test):
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Ham", "Spam"])
        disp.plot(cmap="Blues", values_format="d")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title("Confusion Matrix")
        plt.show()
        plt.savefig('static/img/confussion_matrix.png')
        plt.close()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        return accuracy, precision, recall, f1

        
