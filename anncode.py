import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

# loads in the data from a csv format
file_path = r'C:\Users\Rose\Documents\uni\Diabetes Dataset\training.csv'
data = pd.read_csv(file_path)

# just checking data loaded in correctly
print("Check of the dataset:")
print(data.head())

# converts both the gender and the class columns to numeric form to make it easier for the nn
genders = LabelEncoder()
data['Gender'] = genders.fit_transform(data['Gender'])
classes = LabelEncoder()
data['Class'] = classes.fit_transform(data['Class'])

# checks the first few 
print("\nGender column:")
print(data['Gender'].head())
print("\nClass column:")
print(data['Class'].head())

# splits the dataset into features (z) and targets (y) 
z = data.drop(columns=['Class'])
y = data['Class']

# checks that the datatypes of the features are all numeric
print("\nDatatypes of the features:")
print(z.dtypes)

# standardizes the features
scaler = StandardScaler()
z = scaler.fit_transform(z)

# splits the data into training and testing samples, with 20% being used for testing and a random state of 45 for reproducibility 
ztrain, ztest, ytrain, ytest = train_test_split(z, y, test_size=0.2, random_state=45)

# checks to see if the split worked 
print("\nShape of ztrain:", ztrain.shape)
print("Shape of ztest:", ztest.shape)
print("Shape of ytrain:", ytrain.shape)
print("Shape of ytest:", ytest.shape)

# makes use of tensorflow to build the ann
model = Sequential()
# adds two layers of 16 neurons relu activation with the input dimension equal to the number of features 
model.add(Dense(16, input_dim=ztrain.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
# adds an output layer with the neurons equalling the number of classes
model.add(Dense(len(set(y)), activation='softmax'))

# checks to see the model architecture
print("\nModel Summary:")
model.summary()

# compiles the model 
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# trains the model using the training data. It will be trained 50 times with this dataset
history = model.fit(ztrain, ytrain, epochs=50, batch_size=10, validation_split=0.2, verbose=1)

# evaluates the model based on the loss value and its accuracy 
loss, accuracy = model.evaluate(ztest, ytest, verbose=0)

# outputs the loss value and accuracy
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# makes predictions of the class labels using argmax 
predictions = model.predict(ztest)
class_predictions = predictions.argmax(axis=1)

# outputs the first few predictions to check them and the actual values side by side
print("\nFirst few predictions:")
print(class_predictions[:10])
print("First few actual values:")
print(ytest[:10].values)

# makes a confusion matrix based off the actual values and the predictions
conf_matrix = confusion_matrix(ytest, class_predictions)

# outputs the confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# makes and outputs a detailed classification report
class_rep = classification_report(ytest, class_predictions)
print("\nClassification Report:")
print(class_rep)

# calculates and prints the overall accuracy
overall_accuracy = accuracy_score(ytest, class_predictions)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")