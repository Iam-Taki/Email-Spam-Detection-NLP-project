# Email-Spam-Detection-NLP-project


# Spam Detection Using NLTK and FFNN 

We are using a email classfication dataset with 3002 Column and 5112 rows of data.
We are using Natrual Language Toolkit and FFNN to train a model to detect is the Email is SPAM or not. 

Mission of this project is to learn and train a model to help people to detect the SPAM Mail. So they can safe out from virus and malware and other spam thing.


## Author

- [Iam-Taki](https://github.com/Iam-Taki)
- Abdulllah Al Noman Taki 
- Master's in Artificial Intelligence
- Matricola : VR528988


## Information Section

[Download Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

## Documentation

In Both Model We have use the 95% of the data from dataset. and 5% of data for testing.

## FFNN

* Importing Packages
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer



```

## Approach 1

* NLTK Stopword Setup and Filtering
```python
nltk.download('stopwords')

stop_words = stopwords.words('english')

print(stop_words)

dataset = pd.read_csv('/content/drive/MyDrive/CSV files/emails.csv')
df_main = dataset.drop(columns=["Email No.", "Prediction"])

print(df_main.sample(20))

type(stop_words)

filter = [item for item in df_main.columns if item not in stop_words]

print(filter)
print(len(df_columns))
print(len(filter))

dataset = pd.read_csv('/content/drive/MyDrive/CSV files/emails.csv')
df = pd.DataFrame(dataset, columns=filter)
print(len(df.columns))
```

* Setup the dataset in the directory of the code file.
```python
dataset = pd.read_csv('./emails.csv')
```

you can change the dataset from here.

* I am assigning 2 variable to design the dataset | Main_dataset variable I am dropping 1st and last column. Main_predication_dataset varible the predication.

```python
Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]
```

* Traning the Model and Using 100 Epochs and Using 20 batch Size.
```python
Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]

main_dataset_train, main_dataset_test, main_predication_train, main_predication_test = train_test_split(Main_dataset, Main_predication_dataset, test_size=0.05, random_state=101)

scaler = StandardScaler()
Main_train = scaler.fit_transform(main_dataset_train)
Main_test = scaler.transform(main_dataset_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(Main_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(Main_train, main_predication_train, epochs=20, batch_size=32, validation_data=(Main_test, main_predication_test), verbose=1)

main_trained = (model.predict(Main_test) > 0.5).astype(int)

conf_matrix = confusion_matrix(main_predication_test, main_trained)
f1 = f1_score(main_predication_test, main_trained)
class_report = classification_report(main_predication_test, main_trained)

print("Confusion Matrix:")
print(conf_matrix)

print("\nF1 Score:", f1)

print("\nClassification Report:")
print(class_report)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

![Epoch] Screenshot 2025-06-18 104804.png


![image alt](https://github.com/Iam-Taki/Email-Spam-Detection-NLP-project/blob/0eb1a8e0189a70e0f5df68e1fcdf9ab9d05f2ac6/Screenshot%202025-06-18%20104822.png)

![Validation Map]https://github.com/Iam-Taki/Email-Spam-Detection-NLP-project/blob/0eb1a8e0189a70e0f5df68e1fcdf9ab9d05f2ac6/Screenshot%202025-06-18%20104847.png


## Approach 2 Using LSTM

* NLTK Stopword Setup and Filtering
```python
nltk.download('stopwords')

stop_words = stopwords.words('english')

print(stop_words)

dataset = pd.read_csv('/content/drive/MyDrive/CSV files/emails.csv')
df_main = dataset.drop(columns=["Email No.", "Prediction"])

print(df_main.sample(20))

type(stop_words)

filter = [item for item in df_main.columns if item not in stop_words]

print(filter)
# Changed df_columns to df_main.columns
print(len(df_main.columns))
print(len(filter))

dataset = pd.read_csv('/content/drive/MyDrive/CSV files/emails.csv')
df = pd.DataFrame(dataset, columns=filter)
print(len(df.columns))
```

* Setup the dataset in the directory of the code file.
```python
dataset = pd.read_csv('/content/drive/MyDrive/CSV files/emails.csv')
```

you can change the dataset from here.

* I am assigning 2 variable to design the dataset | Main_dataset variable I am dropping 1st and last column. Main_predication_dataset varible the predication.

```python
Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]
```

* Traning the Model and Using 100 Epochs and Using 30 batch Size.
```python
from tensorflow.keras.layers import Dense, Softmax, Embedding, Bidirectional, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


max_words = 10000
max_len = 200


Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]



main_dataset_train, main_dataset_test, main_predication_train, main_predication_test = train_test_split(Main_dataset, Main_predication_dataset, test_size=0.2, random_state=56)

scaler = StandardScaler()
Main_train = scaler.fit_transform(main_dataset_train)
Main_test = scaler.transform(main_dataset_test)


model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=Main_train.shape[1]),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(Main_train, main_predication_train, epochs=10, batch_size=30,
                   validation_data=(Main_test, main_predication_test), verbose=1)

main_trained = (model.predict(Main_test) > 0.5).astype(int)
```

![Epoch]https://github.com/Iam-Taki/Email-Spam-Detection-NLP-project/blob/05d4ad5fa8d6c0db808aa44463a6010d8e8558d1/Screenshot%202025-06-18%20125535.png

![Confisuon Martrix, F1]https://github.com/Iam-Taki/Email-Spam-Detection-NLP-project/blob/67afbd7f388f38d17b7c0a49f9514e500c4c2964/Screenshot%202025-06-18%20125749.png

![Validation Map]https://github.com/Iam-Taki/Email-Spam-Detection-NLP-project/blob/17d9e8f274df2e0a49edfac6ddce68693f9f3fb2/Screenshot%202025-06-18%20125926.png
