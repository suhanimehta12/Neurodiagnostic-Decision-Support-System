import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.optimizers import Adamax
from PIL import Image
import streamlit as st
from matplotlib import pyplot as plt
import io
import random
import plotly.graph_objects as go

# Function to preprocess the image
def preprocess_image(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# Function to render images using Matplotlib and convert them for Streamlit
def render_image(title, img, color_type):
    plt.figure(figsize=(10, 10))
    if color_type == 'bgr':
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
    elif color_type == 'gray':
        plt.imshow(img, cmap='gray')
    elif color_type == 'rgb':
        plt.imshow(img)
    else:
        raise ValueError("Invalid color type specified.")
    plt.axis('off')
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return Image.open(buf)

# Function to perform segmentation using Otsu's thresholding
def otsu_threshold_segmentation(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return thresh

# Function to remove noise using morphological operations
def remove_noise(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

# Function to perform advanced segmentation using watershed
def watershed_segmentation(image, gray_image):
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(remove_noise(image), kernel, iterations=3)
    dist_transform = cv2.distanceTransform(remove_noise(image), cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    colored_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(colored_image, markers)

    segmented_image = np.zeros_like(colored_image)
    for label in np.unique(markers):
        if label == -1:
            segmented_image[markers == label] = [255, 0, 0]
        elif label > 0:
            random_color = np.random.randint(0, 255, size=3).tolist()
            segmented_image[markers == label] = random_color

    return markers, segmented_image

# Function to calculate tumor volume
def calculate_tumor_volume(image, pixel_spacing, slice_thickness):
    preprocessed_img = preprocess_image(image)
    otsu_thresh = otsu_threshold_segmentation(preprocessed_img)
    cleaned_image = remove_noise(otsu_thresh)
    _, watershed_segmented = watershed_segmentation(cleaned_image, preprocessed_img)
    tumor_pixels = np.sum(cleaned_image == 255)

    if tumor_pixels == 0:
        return preprocessed_img, otsu_thresh, cleaned_image, watershed_segmented, 0, 0, 0

    pixel_area = pixel_spacing[0] * pixel_spacing[1]
    slice_area = tumor_pixels * pixel_area
    tumor_volume = slice_area * slice_thickness

    total_pixels = image.size
    total_area = total_pixels * pixel_area
    total_volume = total_area * slice_thickness
    spread_percentage = (tumor_volume / total_volume) * 100

    return preprocessed_img, otsu_thresh, cleaned_image, watershed_segmented, tumor_volume, total_volume, spread_percentage

# Load pre-trained model
@st.cache_resource
def load_model():
    loaded_model = tf.keras.models.load_model(r'D:/Parth/New folder/model/Brain_Tumors_Classifier_efficientnetb0.h5', compile=False)
    loaded_model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return loaded_model

model = load_model()

# Streamlit UI
st.title("Comprehensive Brain Tumor Analysis")

uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_gray = np.array(image.convert('L'))

    pixel_spacing = (0.5, 0.5)
    slice_thickness = 2.0

    preprocessed_img, otsu_thresh, cleaned_image, watershed_segmented, tumor_volume, total_volume, spread_percentage = calculate_tumor_volume(
        img_gray, pixel_spacing, slice_thickness
    )

    st.write("### Tumor Volume and Spread Analysis")
    if tumor_volume == 0:
        st.write("**No tumor detected.**")
    else:
        st.write(f"**Estimated Tumor Volume:** {tumor_volume:.2f} mm³")
        st.write(f"**Total Slice Volume:** {total_volume:.2f} mm³")
        st.write(f"**Percentage Spread:** {spread_percentage:.2f}%")

        fig = go.Figure(data=[go.Pie(
            labels=['Cancer Spread', 'Remaining Healthy'],
            values=[spread_percentage, 100 - spread_percentage],
            hole=0.3,
            hoverinfo='label+percent',
            textinfo='value+percent',
            marker=dict(colors=['#ff6666', '#66b3ff'])
        )])

        fig.update_layout(
            title="Tumor Spread Percentage",
            annotations=[dict(text=f'{spread_percentage:.2f}%', x=0.5, y=0.5, font_size=30, showarrow=False)]
        )

        st.plotly_chart(fig)

    st.write("### Image Processing Steps")
    st.image(render_image("Preprocessed Image", preprocessed_img, 'gray'), caption="Preprocessed Image", use_container_width=True)
    st.image(render_image("Otsu Thresholding", otsu_thresh, 'gray'), caption="Otsu Thresholded Image", use_container_width=True)
    st.image(render_image("Noise-Removed Image", cleaned_image, 'gray'), caption="Filtered Image", use_container_width=True)
    st.image(render_image("Watershed Segmentation", watershed_segmented, 'rgb'), caption="Watershed Segmented Image", use_container_width=True)

    img_resized = image.resize((224, 224))
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    predicted_class = class_labels[np.argmax(prediction)]

    st.write("### Classification Result")
    if predicted_class == "No Tumor":
        st.write("**The model predicts no tumor in the image.**")
    else:
        st.write(f"**Predicted Class:** {predicted_class}")
