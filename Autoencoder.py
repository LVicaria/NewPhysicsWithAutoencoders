import keras
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from keras.layers import Cropping2D

from DataLoader1 import load_data3
x_train, x_test, y_train, y_test = load_data3()

# Define the structure of the autoencoder
input_img = Input(shape=(25, 25, 1))

# Encoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

# Add Cropping2D layer to adjust the output size to match the input size
cropped_decoded = Cropping2D(cropping=((1, 2), (1, 2)))(x)

autoencoder = Model(input_img, cropped_decoded)
autoencoder.summary()

# Compile the autoencoder using Adam optimizer and MSE loss
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Assuming you've split your data and x_train contains only "normal" data
# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True, validation_split=0.1)

reconstructed = autoencoder.predict(x_test)
reconstruction_error = np.mean(np.power(x_test - reconstructed, 2), axis=(1, 2, 3))

# Determine the threshold from the normal training data
# For example, use the 95th percentile of the normal data's reconstruction error
threshold = np.percentile(reconstruction_error, 95)

# Evaluate anomaly detection performance
y_test_binary = np.argmax(y_test, axis=1)

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test_binary, reconstruction_error)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 8))
plt.title('ROC Curve for Anomaly Detection')
plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.2f}')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Visualize the reconstruction error distribution
plt.figure(figsize=(10, 8))
plt.hist(reconstruction_error, bins=50)
plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()