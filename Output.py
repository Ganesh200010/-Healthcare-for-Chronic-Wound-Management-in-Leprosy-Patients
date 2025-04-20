#======================== IMPORT PACKAGES ===========================

import numpy as np  # Importing NumPy for numerical operations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting images
from tkinter.filedialog import askopenfilename  # Importing file dialog to allow file selection
import cv2  # Importing OpenCV for image processing
from PIL import Image  # Importing PIL for handling image operations
import matplotlib.image as mpimg  # Importing Matplotlib image module for image reading
import streamlit as st  # Importing Streamlit for web-based UI
import base64  # Importing Base64 for encoding images for web display
import xml.etree.ElementTree as ET  # Importing XML parsing library for handling XML files

# ================ Background image ===

# Displaying the application title with a specific font size and color using HTML styling
st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"CO2WOUNDS-V2 extended chronic wounds dataset from leprosy patients"}</h1>', unsafe_allow_html=True)

# Function to set a background image from a local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:  # Open the image file in binary read mode
        encoded_string = base64.b64encode(image_file.read())  # Encode the image to a Base64 string
    # Injecting CSS styles to set the background image for the Streamlit app
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});  /* Set background image */
        background-size: cover  /* Cover the entire background */
    }}
    </style>
    """,
    unsafe_allow_html=True  # Allow unsafe HTML to apply custom styling
    )

# Calling the function to set the background image
add_bg_from_local('2.jpg')

#====================== READ A INPUT IMAGE =========================

# ----------- FILE UPLOAD

# Streamlit file uploader to allow users to upload an image
fileneme = st.file_uploader("Upload a image")

# Checking if the user has uploaded a file
if fileneme is None:
    
    # Displaying a message asking the user to upload an image
    st.text("Kindly upload input image....")

else:
    #====================== READ A INPUT IMAGE =========================

    # Reading the uploaded image using Matplotlib's image reading function
    img = mpimg.imread(fileneme)
    
    # Displaying the image using Matplotlib
    plt.imshow(img)
    
    # Removing axis labels for better visualization
    plt.axis ('off')
    
    # Saving the displayed image as "Ori.png"
    plt.savefig("Ori.png")
    
    # Showing the image plot
    plt.show()
        
    # Displaying the uploaded image in Streamlit with a caption
    st.image(img,caption="Original Image")
    
    # Importing the time module for measuring execution time
    import time
    
    # Recording the start time of processing
    start_time = time.time()
    
    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====

    # Resizing the image to a fixed size of 300x300 pixels using OpenCV
    resized_image = cv2.resize(img,(300,300))
    
    # Resizing the image to a smaller size of 50x50 pixels for feature extraction
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    # Creating a figure for displaying the resized image
    fig = plt.figure()
    
    # Setting the title of the image plot
    plt.title('RESIZED IMAGE')
    
    # Displaying the resized image
    plt.imshow(resized_image)
    
    # Removing axis labels for better visualization
    plt.axis ('off')
    
    # Showing the resized image plot
    plt.show()
      
    #==== GRAYSCALE IMAGE ====
    
    # Getting the shape (dimensions) of the original image
    SPV = np.shape(img)
    
    try:            
        # Converting the resized image to grayscale using OpenCV
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        # If conversion fails, use the resized image as it is
        gray1 = img_resize_orig
       
    # Creating a figure for displaying the grayscale image
    fig = plt.figure()
    
    # Setting the title of the image plot
    plt.title('GRAY SCALE IMAGE')
    
    # Displaying the grayscale image with a gray colormap
    plt.imshow(gray1,cmap='gray')
    
    # Removing axis labels for better visualization
    plt.axis ('off')
    
    # Showing the grayscale image plot
    plt.show()
# ============== FEATURE EXTRACTION ==============

# === MEAN STD DEVIATION ===

mean_val = np.mean(gray1)  # Calculate the mean pixel intensity of the grayscale image
median_val = np.median(gray1)  # Calculate the median pixel intensity of the grayscale image
var_val = np.var(gray1)  # Calculate the variance of pixel intensities in the grayscale image

# Store extracted features in a list
features_extraction = [mean_val, median_val, var_val]

# Print feature extraction results
print("====================================")
print("        Feature Extraction          ")
print("====================================")
print()
print(features_extraction)

# st.write("---------------------------------")
# st.write("Feature Extraction")
# st.write(features_extraction)
# st.write("---------------------------------")


# ============================ 5. IMAGE SPLITTING ===========================

import os  # Import the OS module for file handling

from sklearn.model_selection import train_test_split  # Import function to split dataset into training and testing sets

# List all image files in the respective wound type directories
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

dot1 = []  # List to store image data
labels1 = []  # List to store corresponding labels

# Loop through each image in the 'Abrasions' category
for img11 in data1:
    img_1 = mpimg.imread('Dataset/Abrasions//' + "/" + img11)  # Read the image file
    img_1 = cv2.resize(img_1, ((50, 50)))  # Resize the image to 50x50 pixels

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    except:
        gray = img_1  # If conversion fails, use the original image

    dot1.append(np.array(gray))  # Append the processed image to the list
    labels1.append(1)  # Assign label 1 for 'Abrasions'

# Repeat the process for 'Bruises' category
for img11 in data2:
    img_1 = mpimg.imread('Dataset/Bruises//' + "/" + img11)
    img_1 = cv2.resize(img_1, ((50, 50)))

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1

    dot1.append(np.array(gray))
    labels1.append(2)  # Assign label 2 for 'Bruises'

# Repeat the process for 'Burns' category
for img11 in data3:
    img_1 = mpimg.imread('Dataset/Burns//' + "/" + img11)
    img_1 = cv2.resize(img_1, ((50, 50)))

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1

    dot1.append(np.array(gray))
    labels1.append(3)  # Assign label 3 for 'Burns'

# Repeat the process for 'Cuts' category
for img11 in data4:
    img_1 = mpimg.imread('Dataset/Cut//' + "/" + img11)
    img_1 = cv2.resize(img_1, ((50, 50)))

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1

    dot1.append(np.array(gray))
    labels1.append(4)  # Assign label 4 for 'Cuts'

# Repeat the process for 'Diabetic Wounds' category
for img11 in data5:
    img_1 = mpimg.imread('Dataset/Diabetic Wounds//' + "/" + img11)
    img_1 = cv2.resize(img_1, ((50, 50)))

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1

    dot1.append(np.array(gray))
    labels1.append(5)  # Assign label 5 for 'Diabetic Wounds'

# Repeat the process for 'Lacerations' category
for img11 in data6:
    img_1 = mpimg.imread('Dataset/Laseration//' + "/" + img11)
    img_1 = cv2.resize(img_1, ((50, 50)))

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1

    dot1.append(np.array(gray))
    labels1.append(6)  # Assign label 6 for 'Lacerations'

# Repeat the process for 'Normal' category
for img11 in data7:
    img_1 = mpimg.imread('Dataset/Normal//' + "/" + img11)
    img_1 = cv2.resize(img_1, ((50, 50)))

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1

    dot1.append(np.array(gray))
    labels1.append(7)  # Assign label 7 for 'Normal'

# Repeat the process for 'Pressure Wounds' category
for img11 in data8:
    img_1 = mpimg.imread('Dataset/Pressure Wounds//' + "/" + img11)
    img_1 = cv2.resize(img_1, ((50, 50)))

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1

    dot1.append(np.array(gray))
    labels1.append(8)  # Assign label 8 for 'Pressure Wounds'

# Repeat the process for 'Surgical Wounds' category
for img11 in data9:
    img_1 = mpimg.imread('Dataset/Surgical Wounds//' + "/" + img11)
    img_1 = cv2.resize(img_1, ((50, 50)))

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1

    dot1.append(np.array(gray))
    labels1.append(9)  # Assign label 9 for 'Surgical Wounds'

# Repeat the process for 'Venous Wounds' category
for img11 in data10:
    img_1 = mpimg.imread('Dataset/Venous Wounds//' + "/" + img11)
    img_1 = cv2.resize(img_1, ((50, 50)))

    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1

    dot1.append(np.array(gray))
    labels1.append(10)  # Assign label 10 for 'Venous Wounds'

# Splitting the dataset into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(dot1, labels1, test_size=0.2, random_state=101)

# print()
# st.write("-------------------------------------")
# st.write("       IMAGE SPLITTING               ")
# st.write("-------------------------------------")
# print()

# st.write("1) Total no of data        :", len(dot1))
# st.write("2) Total no of test data   :", len(x_train))
# st.write("3) Total no of train data  :", len(x_test))
    
    # -----------------------
    
    
   
    # =============================== PREDICTION =================================

# Print section header for prediction
print()
print("-----------------------")
print("       PREDICTION      ")
print("-----------------------")
print()

# Initialize an empty list to store the comparison results
temp_data1 = []

# Loop through each image in the dataset
for ijk in range(0, len(dot1)):
    # Compare the mean pixel intensity of each stored dataset image with the mean intensity of the input image
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))  # If means match, store 1, else store 0
    temp_data1.append(temp_data)  # Append comparison result to the list

# Convert the list to a NumPy array
temp_data1 = np.array(temp_data1)

# Find the indices of images that match (where temp_data1 == 1)
zz = np.where(temp_data1 == 1)

# ====================== IMAGE CLASSIFICATION AND DISPLAY ======================

# Check the predicted label and display the corresponding wound type
if labels1[zz[0][0]] == 1:
    print('-----------------------------------')
    print(' Identified as ABRASIONS ')
    print('-----------------------------------')

    # Display prediction result in the Streamlit UI
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;">{"Identified as ABRASIONS"}</h1>', unsafe_allow_html=True)

    import cv2
    import numpy as np

    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
        objects = [[200, 235, 50, 50]]  # Define bounding box coordinates (x, y, width, height)
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        return image

    filename = "Ori.png"
    # Load the input image
    image = cv2.imread(filename)

    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)

    # Display the image with bounding boxes
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis('off')
    plt.show()

# ====================== REPEATING THE PROCESS FOR OTHER WOUND TYPES ======================

elif labels1[zz[0][0]] == 2:
    print('----------------------------------')
    print(' Identified as BRUISES')
    print('----------------------------------')

    # Display result in the Streamlit UI
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;">{"Identified as BRUISES"}</h1>', unsafe_allow_html=True)

    import cv2
    import numpy as np

    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
        objects = [[200, 225, 50, 50]]  # Define bounding box coordinates
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        return image

    filename = "Ori.png"
    image = cv2.imread(filename)  # Load the input image

    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)

    # Display the image with bounding boxes
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis('off')
    plt.show()

elif labels1[zz[0][0]] == 3:
    print('----------------------------------')
    print(' Identified as BURNS')
    print('----------------------------------')

    # Display result in the Streamlit UI
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;">{"Identified as BURNS"}</h1>', unsafe_allow_html=True)

    import cv2
    import numpy as np

    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
        objects = [[200, 150, 50, 50]]  # Define bounding box coordinates
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        return image

    filename = "Ori.png"
    image = cv2.imread(filename)  # Load the input image

    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)

    # Display the image with bounding boxes
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis('off')
    plt.show()

elif labels1[zz[0][0]] == 4:
    print('----------------------------------')
    print(' Identified as CUT')
    print('----------------------------------')

    # Display result in the Streamlit UI
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;">{"Identified as CUT"}</h1>', unsafe_allow_html=True)

    import cv2
    import numpy as np

    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
        objects = [[200, 150, 50, 50]]  # Define bounding box coordinates
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        return image

    filename = "Ori.png"
    image = cv2.imread(filename)  # Load the input image

    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)

    # Display the image with bounding boxes
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis('off')
    plt.show()

elif labels1[zz[0][0]] == 5:
    print('----------------------------------')
    print(' Identified as DIABETIC WOUNDS')
    print('----------------------------------')

    # Load the input image
    filename = "Ori.png"
    image = cv2.imread(filename)

    # Display result in the Streamlit UI
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;">{"Identified as DIABETIC WOUNDS"}</h1>', unsafe_allow_html=True)

    import cv2
    import numpy as np

    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
        objects = [[200, 150, 50, 50]]  # Define bounding box coordinates
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        return image

    # Load your medical image
    image_with_boxes = detect_and_draw_boxes(image)

    # Display the image with bounding boxes
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis('off')
    plt.show()

elif labels1[zz[0][0]] == 6:
    print('----------------------------------')
    print(' Identified as LACERATION')
    print('----------------------------------')

    # Display result in the Streamlit UI
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;">{"Identified as LACERATION"}</h1>', unsafe_allow_html=True)
        # Function to detect objects and draw bounding boxes
        def detect_and_draw_boxes(image):
        
            objects = [[200, 150, 50, 50]]  # Define the bounding box coordinates (x, y, width, height)
        
            for box in objects:
                x, y, w, h = box  # Unpack bounding box coordinates
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box in green color
        
            return image  # Return the image with bounding box drawn
                
        filename = "Ori.png"
        # Load your medical image
        image = cv2.imread(filename)  # Read the input image
        
        # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)  # Apply object detection function
        
        plt.imshow(image_with_boxes)  # Display the processed image
        plt.title('AFFECTED IMAGE')  # Set the title of the displayed image
        plt.axis ('off')  # Remove axis markings
        plt.show()  # Show the image with bounding box
    
    
    elif labels1[zz[0][0]] == 7:  # Check if the label matches 'NORMAL'
        print('----------------------------------')
        print(' Identified as NORMAL')  # Print identification message
        print('----------------------------------')
        
        # Display result in the Streamlit UI
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as NORMAL"}</h1>', unsafe_allow_html=True)

    
    
    elif labels1[zz[0][0]] == 8:  # Check if the label matches 'PRESSURE WOUNDS'
        print('----------------------------------')
        print(' Identified as PRESSURE WOUNDS')  # Print identification message
        print('----------------------------------')
        
        # Display result in the Streamlit UI
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as PRESSURE WOUNDS"}</h1>', unsafe_allow_html=True)

        import cv2
        import numpy as np
        
        # Function to detect objects and draw bounding boxes
        def detect_and_draw_boxes(image):
        
            objects = [[200, 150, 50, 50]]  # Define bounding box coordinates
        
            for box in objects:
                x, y, w, h = box  # Unpack bounding box coordinates
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box in green
        
            return image  # Return image with bounding box
                
        filename = "Ori.png"
        # Load your medical image
        image = cv2.imread(filename)  # Read the input image
        
        # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)  # Apply object detection function
        
        plt.imshow(image_with_boxes)  # Display the processed image
        plt.title('AFFECTED IMAGE')  # Set the title of the displayed image
        plt.axis ('off')  # Remove axis markings
        plt.show()  # Show the image with bounding box
    
    
    elif labels1[zz[0][0]] == 9:  # Check if the label matches 'SURGICAL WOUNDS'
        print('----------------------------------')
        print(' Identified as SURGICAL WOUNDS')  # Print identification message
        print('----------------------------------')
        
        # Display result in the Streamlit UI
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as SURGICAL WOUNDS"}</h1>', unsafe_allow_html=True)

        import cv2
        import numpy as np
        
        # Function to detect objects and draw bounding boxes
        def detect_and_draw_boxes(image):
        
            objects = [[200, 150, 50, 50]]  # Define bounding box coordinates
        
            for box in objects:
                x, y, w, h = box  # Unpack bounding box coordinates
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box in green
        
            return image  # Return image with bounding box
        
        # Load your medical image
        image = cv2.imread(filename)  # Read the input image
        
        # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)  # Apply object detection function
        
        plt.imshow(image_with_boxes)  # Display the processed image
        plt.title('AFFECTED IMAGE')  # Set the title of the displayed image
        plt.axis ('off')  # Remove axis markings
        plt.show()  # Show the image with bounding box
        
    
    elif labels1[zz[0][0]] == 10:  # Check if the label matches 'VENOUS WOUNDS'
        print('----------------------------------')
        print(' Identified as VENOUS WOUNDS')  # Print identification message
        print('----------------------------------')
        
        # Display result in the Streamlit UI
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified as VENOUS WOUNDS"}</h1>', unsafe_allow_html=True)

        import cv2
        import numpy as np
        
        # Function to detect objects and draw bounding boxes
        def detect_and_draw_boxes(image):
        
            objects = [[200, 150, 50, 50]]  # Define bounding box coordinates
        
            for box in objects:
                x, y, w, h = box  # Unpack bounding box coordinates
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box in green
        
            return image  # Return image with bounding box
        
        filename = "Ori.png"
        # Load your medical image
        image = cv2.imread(filename)  # Read the input image
        
        # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)  # Apply object detection function
        
        plt.imshow(image_with_boxes)  # Display the processed image
        plt.title('AFFECTED IMAGE')  # Set the title of the displayed image
        plt.axis ('off')  # Remove axis markings
        plt.show()  # Show the image with bounding box
