# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import tensorflow as tf

# Getting the dataset
dataset = pd.read_csv("test.csv")

# Getting the essential calues
# All are type NumPy array (_s1 means scenario 1 or A)
passenger_id_s1 = dataset["PassengerId"].values
p_class_s1 = dataset["Pclass"].values
sex_s1 = dataset["Sex"].values
age_s1 = dataset["Age"].values
sib_sp_s1 = dataset["SibSp"].values
par_ch_s1 = dataset["Parch"].values
cabin_no_s1 = dataset["Cabin"].values

# Encoding the Sex
lbl_encoder_s1 = LabelEncoder()
sex_s1 = lbl_encoder_s1.fit_transform(sex_s1)
# Imputing the missing values
cabin_no_s1 = cabin_no_s1.reshape(-1, 1)
imputer_s1 = SimpleImputer(strategy = "constant", fill_value = "XXX")
imputer_age_s1 = SimpleImputer(strategy="median")
cabin_no_s1 = imputer_s1.fit_transform(cabin_no_s1)
age_s1 = imputer_age_s1.fit_transform(age_s1.reshape(-1, 1))
# Encoding & Imputing the Cabin No
lbl_encoder_cabin_no_s1 = LabelEncoder()
impute_cabin_no_s1 = SimpleImputer(strategy = "constant", fill_value = "XXX")
cabin_no_s1 = impute_cabin_no_s1.fit_transform(cabin_no_s1)
cabin_no_s1 = lbl_encoder_cabin_no_s1.fit_transform(cabin_no_s1.ravel())

# Reshaping
passenger_id_s1 = passenger_id_s1.reshape(-1, 1)
p_class_s1 = p_class_s1.reshape(-1, 1)
sex_s1 = sex_s1.reshape(-1, 1)
age_s1 = age_s1.reshape(-1, 1)
sib_sp_s1 = sib_sp_s1.reshape(-1, 1)
par_ch_s1 = par_ch_s1.reshape(-1, 1)
cabin_no_s1 = cabin_no_s1.reshape(-1, 1)
# Merging 
input_s1 = np.concatenate((passenger_id_s1, p_class_s1, sex_s1, age_s1, sib_sp_s1, par_ch_s1, cabin_no_s1), axis=1)

# Importing the Scaler
from sklearn.preprocessing import StandardScaler
scaler_s1 = StandardScaler()
input_s1 = scaler_s1.fit_transform(input_s1)

# Defining the neural network
nn = tf.keras.models.load_model("true_model")
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
pred = nn.predict(input_s1)

# Making the values between (0-1)

for i in range(0, len(pred)):
  if pred[i] >= 0.7:
    pred[i] = 1
  else:
    pred[i] = 0

# Editing into an excel sheet

# Opening an excel sheet
submission_file = open("submissions1.csv", 'a')

# Writing into file
for j in range (0, 418):
    content_to_write = str(int(passenger_id_s1[j]))+","+str(int(pred[j]))+"\n"
    submission_file.write(content_to_write)

# Closing a file
submission_file.close()