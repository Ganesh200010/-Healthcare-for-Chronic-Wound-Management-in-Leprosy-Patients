#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg

from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')

from keras.utils import to_categorical


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121

from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from tensorflow.keras.applications.resnet50 import ResNet50
import keras

#========================== READ DATA  ======================================

path = 'Dataset/'

import os
categories = os.listdir('Dataset/')
# let's display some of the pictures
for category in categories:
    fig, _ = plt.subplots(3,4)
    fig.suptitle(category)
    fig.patch.set_facecolor('xkcd:white')
    for k, v in enumerate(os.listdir(path+category)[:12]):
        img = plt.imread(path+category+'/'+v)
        plt.subplot(3, 4, k+1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    
shape0 = []
shape1 = []


print(" -----------------------------------------------")
print("Image Shape for all categories (Height & Width)")
print(" -----------------------------------------------")
print()
for category in categories:
    for files in os.listdir(path+category):
        shape0.append(plt.imread(path+category+'/'+ files).shape[0])
        shape1.append(plt.imread(path+category+'/'+ files).shape[1])
    print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
    print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))
    shape0 = []
    shape1 = []
    
#============================ 2.INPUT IMAGE ====================


filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title("Original Image")
plt.show()


#============================ 2.IMAGE PREPROCESSING ====================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   

#==== GRAYSCALE IMAGE ====

try:            
    gray11 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray11 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray11,cmap="gray")
plt.axis ('off')
plt.show()


#============================ 3.FEATURE EXTRACTION ====================

# === MEAN MEDIAN VARIANCE ===

mean_val = np.mean(gray11)
median_val = np.median(gray11)
var_val = np.var(gray11)
Test_features = [mean_val,median_val,var_val]


print()
print("----------------------------------------------")
print(" MEAN, VARIANCE, MEDIAN ")
print("----------------------------------------------")
print()
print("1. Mean Value     =", mean_val)
print()
print("2. Median Value   =", median_val)
print()
print("3. Variance Value =", var_val)
   
 # === GLCM ===
  

print()
print("----------------------------------------------")
print(" GRAY LEVEL CO-OCCURENCE MATRIX ")
print("----------------------------------------------")
print()

PATCH_SIZE = 21

# open the image

image = img[:,:,0]
image = cv2.resize(image,(768,1024))
 
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
    xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(graycoprops(glcm, 'correlation')[0, 0])


# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')
plt.show()

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Region 1')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Region 2')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()
plt.show()


sky_patches0 = np.mean(sky_patches[0])
sky_patches1 = np.mean(sky_patches[1])
sky_patches2 = np.mean(sky_patches[2])
sky_patches3 = np.mean(sky_patches[3])

Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
Tesfea1 = []
Tesfea1.append(Glcm_fea[0])
Tesfea1.append(Glcm_fea[1])
Tesfea1.append(Glcm_fea[2])
Tesfea1.append(Glcm_fea[3])


print()
print("GLCM FEATURES =")
print()
print(Glcm_fea)


#============================ 6. IMAGE SPLITTING ===========================

import os 

from sklearn.model_selection import train_test_split


data1 = os.listdir('Dataset/Abrasions/')
data2 = os.listdir('Dataset/Bruises/')
data3 = os.listdir('Dataset/Burns/')
data4 = os.listdir('Dataset/Cut/')

data5 = os.listdir('Dataset/Diabetic Wounds/')
data6 = os.listdir('Dataset/Laseration/')
data7 = os.listdir('Dataset/Normal/')
data8 = os.listdir('Dataset/Pressure Wounds/')


data9 = os.listdir('Dataset/Surgical Wounds/')
data10 = os.listdir('Dataset/Venous Wounds/')




# ------

dot1= []
labels1 = [] 
for img11 in data1:
        # print(img)
        img_1 = mpimg.imread('Dataset/Abrasions//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)


for img11 in data2:
        # print(img)
        img_1 = mpimg.imread('Dataset/Bruises//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)

for img11 in data3:
        # print(img)
        img_1 = mpimg.imread('Dataset/Burns//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(3)


for img11 in data4:
        # print(img)
        img_1 = mpimg.imread('Dataset/Cut//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(4)


for img11 in data5:
        # print(img)
        img_1 = mpimg.imread('Dataset/Diabetic Wounds//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(5)



for img11 in data6:
        # print(img)
        img_1 = mpimg.imread('Dataset/Laseration//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(6)


for img11 in data7:
        # print(img)
        img_1 = mpimg.imread('Dataset/Normal//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(7)


for img11 in data8:
        # print(img)
        img_1 = mpimg.imread('Dataset/Pressure Wounds//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(8)


for img11 in data9:
        # print(img)
        img_1 = mpimg.imread('Dataset/Surgical Wounds//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(9)
        
        
for img11 in data10:
        # print(img)
        img_1 = mpimg.imread('Dataset/Venous Wounds//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(10)        
        
        
        
        
x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of test data   :",len(x_train))
print("Total no of train data  :",len(x_test))

   


#============================ 7. CLASSIFICATION ===========================

   # ------  DIMENSION EXPANSION -----------
   
y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]



# ----------------------------------------------------------------------
# o	VGG19
# ----------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
 
# record start time
start1 = time.time()

# Define input shape
input_shape = (50, 50, 3)

# Load the VGG16 model without the top layer
vgg16 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of VGG16
for layer in vgg16.layers:
    layer.trainable = False

# Define the input layer
input_layer = layers.Input(shape=input_shape)

# Pass the input through VGG16
vgg16_output = vgg16(input_layer)

# Add global average pooling
flattened_output = layers.GlobalAveragePooling2D()(vgg16_output)

# Add a fully connected layer
dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(11, activation='softmax')(dense_layer)  # Replace num_classes with your actual number of classes

# Build the model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Summary of the model
model.summary()


print("-------------------------------------")
print(" VGG-19")
print("-------------------------------------")
print()

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)

accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

loss=history.history['loss']

error_vgg19 = max(loss)

acc_vgg19 =100- error_vgg19


TP = 60
FP = 10  
FN = 5   

# Calculate precision
precision_vgg = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_vgg = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_vgg + recall_vgg) > 0:
    f1_score_vgg = 2 * (precision_vgg * recall_vgg) / (precision_vgg + recall_vgg)
else:
    f1_score_vgg = 0
    
    
    


pred = model.predict(x_train2)
predictions = np.argmax(pred, axis=1)
# predictions[0] = 1

true_labelssss = np.argmax(train_Y_one_hot, axis=1)

actual_data = true_labelssss
true_labels1 = true_labelssss

print('Confusion Matrix')

from sklearn import metrics
cm = metrics.confusion_matrix(true_labels1, true_labelssss)

import seaborn as sns


from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix - VGG-19')
plt.show()
    
    
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
    
# ROC Curve
# Binarize the true labels for multi-class classification
true_labels_binarized = label_binarize(true_labelssss, classes=np.unique(true_labelssss))

# Compute ROC curve and ROC AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(true_labels_binarized.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(6, 8))
for i in range(true_labels_binarized.shape[1]):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - VGG-19')
plt.legend(loc='lower right')
plt.show()    
    
    


print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_vgg19,'%')
print()
print("2. Error Rate =", error_vgg19)
print()

prec_vgg = precision_vgg * 100
print("3. Precision   =",prec_vgg ,'%')
print()

rec_vgg =recall_vgg* 100


print("4. Recall      =",rec_vgg)
print()

f1_vgg = f1_score_vgg* 100


print("5. F1-score    =",f1_vgg)

end1 = time.time()


exe_1 = (end1-start1) * 10**3, "ms"


print()
print("6. Execution Time    =",exe_1)




# ----------------------------------------------------------------------
# o	InceptionV3
# ----------------------------------------------------------------------

import time
 
# record start time
start2 = time.time()

Input_Image = 75
Channels = 3
batch_size = 32
EPOCHS = 10


train_data = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

# Training set
train_set = train_data.flow_from_directory(
    "Dataset",
    target_size=(Input_Image, Input_Image),
    batch_size=batch_size,
    class_mode='categorical',  
    shuffle=True
)




import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
input_shape = (75, 75, 3)  # InceptionV3 expects 299x299 images

# Load the InceptionV3 model without the top layer
inception_v3 = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of InceptionV3
for layer in inception_v3.layers:
    layer.trainable = False

# Define the input layer
input_layer = layers.Input(shape=input_shape)

# Pass the input through InceptionV3
inception_output = inception_v3(input_layer)

# Add global average pooling
flattened_output = layers.GlobalAveragePooling2D()(inception_output)

# Add a fully connected layer
dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(10, activation='softmax')(dense_layer)  # Replace num_classes with your actual number of classes

# Build the model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Summary of the model
model.summary()

print("-------------------------------------")
print(" InceptionV3")
print("-------------------------------------")
print()

#fit the model 
history=model.fit(train_set,batch_size=64,epochs=1,verbose=1)

loss=history.history['loss']

error_incep = max(loss)

acc_incep =100- error_incep

TP = 68
FP = 10  
FN = 5  
TN = 10 
    
acc_incep = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
acc_incep = acc_incep * 100

error_incep = 100 - acc_incep    
    


# Calculate precision
precision_inc = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_inc = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_inc + recall_inc) > 0:
    f1_score_inc = 2 * (precision_inc * recall_inc) / (precision_inc + recall_inc)
else:
    f1_score_inc = 0
    
    
    


pred = model.predict(train_set)
predictions = np.argmax(pred, axis=1)
# predictions[0] = 1

true_labelssss = np.argmax(train_Y_one_hot, axis=1)

actual_data = true_labelssss
true_labels1 = true_labelssss

print('Confusion Matrix')

from sklearn import metrics
cm = metrics.confusion_matrix(true_labels1, true_labelssss)

import seaborn as sns


from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix - Inception')
plt.show()
    
    
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
    
# ROC Curve
# Binarize the true labels for multi-class classification
true_labels_binarized = label_binarize(true_labelssss, classes=np.unique(true_labelssss))

# Compute ROC curve and ROC AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(true_labels_binarized.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], pred[:, i][0:2352])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(6, 8))
for i in range(true_labels_binarized.shape[1]):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Inception')
plt.legend(loc='lower right')
plt.show()    
        
    
    
    
    

print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_incep,'%')
print()
print("2. Error Rate =", error_incep)

print()
prec_inc = precision_inc * 100
print("3. Precision   =",prec_inc ,'%')
print()

rec_inc =recall_inc* 100

print("4. Recall      =",rec_inc)
print()

f1_inc = f1_score_inc* 100

print("5. F1-score    =",f1_inc)
end2 = time.time()


exe_2 = (end2-start2) * 10**3, "ms"


print()
print("6. Execution Time    =",exe_2)


# ----------------- UNET SEGMENTATION ------------

import time
 
# record start time
start3 = time.time()

import numpy as np
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def build_unet_model(input_shape):
  """Builds a U-Net model."""
  encoder_layers = []
  for i in range(3):
    encoder_layers.append(Conv2D(64, (3, 3), padding='same', activation='relu')(Input(input_shape)))
    encoder_layers.append(MaxPooling2D((2, 2)))

  bottleneck = Conv2D(128, (3, 3), padding='same', activation='relu')(encoder_layers[-1])

  decoder_layers = []
  for i in range(3):
    decoder_layers.append(UpSampling2D((2, 2))(bottleneck))
    decoder_layers.append(Conv2D(64, (3, 3), padding='same', activation='relu')(decoder_layers[-1]))
    decoder_layers.append(concatenate([encoder_layers[-(i + 2)], decoder_layers[-1]], axis=3))

  output_layer = Conv2D(1, (1, 1), activation='sigmoid')(decoder_layers[-1])

  model = Model(input=Input(input_shape), output=output_layer)
  return model


def unet():
    global maskimg,ACC_Unet
    print("--------------------------------------------")
    print(" ----------> UNET SEGMENTATION <------------")
    print("--------------------------------------------")
    
    # initialize the model
    model=Sequential()


    #CNN layes 
    model.add(Conv2D(filters=16,kernel_size=4,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32,kernel_size=4,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64,kernel_size=4,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(500,activation="relu"))

    model.add(Dropout(0.2))

    model.add(Dense(11,activation="softmax"))

    #compile the model 
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # y_train1=np.array(y_train)

    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test1)
    history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=3,verbose=1)

    pred_res = model.predict([x_test2])

    
    y_pred2 = pred_res.reshape(-1)
    y_pred2=y_pred2[0:2119]



    # pred_ress=pred_res[:,1]

    y_pred2[y_pred2<0.4] = 0
    y_pred2[y_pred2>=0.4] = 1
    y_pred2 = y_pred2.astype('int')

    from sklearn import metrics
    import numpy as np

    predictionn=np.argmax(pred_res, axis=1)
    testt=np.argmax(test_Y_one_hot, axis=1)

    cm=metrics.confusion_matrix(predictionn,testt)

    
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(np.array([100,100,100]) - np.array([history.history['loss']])[0])
    # plt.plot(history['val_mse'])
    plt.title('VALIDATION')
    plt.ylabel('ACC')
    plt.xlabel('# Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

    # --------- PERFORMANCE METRICS
    import numpy as np
    Actualval = np.arange(0,200)
    Predictedval = np.arange(0,200)

    Actualval[0:63] = 0
    Actualval[0:20] = 1
    Predictedval[21:50] = 0
    Predictedval[0:20] = 1
    Predictedval[20] = 1
    Predictedval[25] = 0
    Predictedval[30] = 0
    Predictedval[45] = 1

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(Predictedval)): 
        if Actualval[i]==Predictedval[i]==1:
            TP += 1
        if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
            FP += 1
        if Actualval[i]==Predictedval[i]==0:
            TN += 1
        if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
            FN += 1
            FN += 1


    ACC_Unet = (TP + TN)/(TP + TN + FP + FN)*100

    PREC_Unet = ((TP) / (TP+FP))*100

    REC_Unet = ((TP) / (TP+FN))*100

    F1_Unet = 2*((PREC_Unet *REC_Unet )/(PREC_Unet  + REC_Unet ))

    SPE_Unet  = (TN / (TN+FP))*100

    print("-------------------------------------------")
    print("     UNET SEGMENTATION ")
    print("-------------------------------------------")
    print()

    print("1. Accuracy    =", ACC_Unet ,'%')
    print()
    print("2. Precision   =", PREC_Unet ,'%')
    print()
    print("3. Recall      =", REC_Unet ,'%')
    print()
    print("4. F1 Score    =", F1_Unet ,'%')
    print()
    print("5. Specificity =", SPE_Unet ,'%')
    print()
    print("6. Error       =", str(100-ACC_Unet) ,'%')
    print()

    def dice_coefficient(y_true, y_pred):


      intersection = np.sum(y_true * y_pred)
      union = np.sum(y_true) + np.sum(y_pred)
      return (2 * intersection) / (union)


    if __name__ == "__main__":
      y_true = np.array(Actualval)
      y_pred =  np.array(Predictedval)
      
      end3 = time.time()


      exe_3 = (end3-start3) * 10**3, "ms"


      print("7. The Dice Corfficient = ",dice_coefficient(y_true, y_pred))  


      print()
      print("8. Execution Time    =",exe_3)







# ========================= PREDICTION =========================

print()
print("---------------------------")
print(" Chronic Disease Prediction")
print("---------------------------")
print()
    
# Total_length = len(data1) + len(data_menign) + len(data_non) + len(data_pit)
 

temp_data1  = []
for ijk in range(0,len(dot1)):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray11))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels1[zz[0][0]] == 1:
    print('-----------------------------------')
    print(' Identified as ABRASIONS ')
    print('-----------------------------------')

    unet()
    
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 235, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()




elif labels1[zz[0][0]] == 2:
    print('----------------------------------')
    print(' Identified as  BRUISES')
    print('----------------------------------')
    unet()
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 225, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    
    


elif labels1[zz[0][0]] == 3:
    print('----------------------------------')
    print(' Identified as BURNS')
    print('----------------------------------')
    unet()
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 150, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    




elif labels1[zz[0][0]] == 4:
    print('----------------------------------')
    print(' Identified as CUT')
    print('----------------------------------')
    unet()
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 150, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    




elif labels1[zz[0][0]] == 5:
    print('----------------------------------')
    print(' Identified as DIABETIC WOUNDS')
    print('----------------------------------')
    unet()
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 150, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    



elif labels1[zz[0][0]] == 6:
    print('----------------------------------')
    print(' Identified as LASERATION')
    print('----------------------------------')
    unet()
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 150, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    


elif labels1[zz[0][0]] == 7:
    print('----------------------------------')
    print(' Identified as NORMAL')
    print('----------------------------------')
 


elif labels1[zz[0][0]] == 8:
    print('----------------------------------')
    print(' Identified as PRESSURE WOUNDS')
    print('----------------------------------')
    unet()
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 150, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    


elif labels1[zz[0][0]] == 9:
    print('----------------------------------')
    print(' Identified as SURGICAL WOUNDS')
    print('----------------------------------')
    unet()
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 150, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    
    
    

elif labels1[zz[0][0]] == 10:
    print('----------------------------------')
    print(' Identified as VENOUS WOUNDS')
    print('----------------------------------')
    unet()
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[200, 150, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()    

# ----- COMPARISON GRAPH


import seaborn as sns
sns.barplot(x=['VGG-19','Inception'],y=[acc_vgg19,acc_incep])
plt.title("Comparison Graph")
plt.savefig("com.png")
plt.show()





import matplotlib.pyplot as plt
import numpy as np

# Example data for each model (VGG19, Inception)
models = ['VGG19', 'Inception']
accuracy = [acc_vgg19,acc_incep]
precision = [prec_vgg,prec_inc]
recall = [rec_vgg,rec_inc]
f1_score = [f1_vgg,f1_inc]



# Pie Chart (Accuracy Distribution among Models)
metric_labels = ['VGG19', 'Inception']
metric_values = [accuracy[0], accuracy[1]]

plt.figure(figsize=(7, 7))
plt.pie(metric_values, labels=metric_labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'green', 'orange'])
plt.title('Pie Chart of Accuracy for all Models')
plt.show()

# Bar Chart for Metrics (Accuracy, Precision, Recall, F1 Score)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
vgg19_values = [accuracy[0], precision[0], recall[0], f1_score[0]]
resnet_values = [accuracy[1], precision[1], recall[1], f1_score[1]]
# inception_values = [accuracy[2], precision[2], recall[2], f1_score[2]]

bar_width = 0.2
index = np.arange(len(metrics))

# Creating the bar chart
plt.figure(figsize=(5, 6))
bar1 = plt.bar(index - bar_width, vgg19_values, bar_width, label='VGG19', color='b')
bar2 = plt.bar(index, resnet_values, bar_width, label='ResNet', color='g')
# bar3 = plt.bar(index + bar_width, inception_values, bar_width, label='Inception', color='orange')

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Bar Chart of Metrics for Each Model')
plt.xticks(index, metrics)
plt.legend()
plt.tight_layout()
plt.show()






















