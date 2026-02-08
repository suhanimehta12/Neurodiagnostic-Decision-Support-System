# Automated Tumor Detection, Classification, and Segmentation in MRI Images

## Project Overview
This project delivers a comprehensive AI-based system for automated tumor detection, classification, and segmentation in MRI images. By combining advanced image processing techniques with deep learning models, the system enhances MRI clarity, accurately identifies tumor boundaries, estimates tumor volume, and calculates the percentage of affected brain tissue. The workflow provides a robust and scalable medical imaging analysis pipeline suitable for research and clinical applications.



## Key Highlights

- Automated Tumor Detection and Classification  
  Detects tumor regions and classifies MRI images with high accuracy to support early diagnosis.

- Advanced Segmentation Pipeline  
  Uses Watershed Segmentation and Morphological Operations to extract precise tumor boundaries.

- Enhanced Image Preprocessing
  Improves contrast, reduces noise, and sharpens MRI features to increase segmentation accuracy.

- Tumor Volume and Affected Region Estimation
  Calculates tumor volume and the percentage of affected brain tissue for quantitative assessment.

- Model Comparison and Optimization
  Evaluates multiple pre-trained CNN architectures, including VGG16 and ResNet, to select the most accurate model.

- Visualization and Interpretability
  Generates overlay images to highlight tumor boundaries on original MRI scans, improving clinical interpretability.


## Technology Stack

Languages: 
- Python

Libraries and Frameworks:
- OpenCV  
- NumPy  
- Matplotlib  
- scikit-image  
- TensorFlow / Keras  
- scikit-learn

Techniques and Algorithms: 
- Image Enhancement: Histogram Equalization, Contrast Stretching  
- Morphological Filtering: Erosion, Dilation, Opening, Closing  
- Watershed Algorithm for Tumor Segmentation  
- Contour Detection for Tumor Localization  
- CNN-based Classification (VGG16, ResNet)  
- Tumor Volume and Percentage Computation  

## Workflow Overview

1. Load MRI Images  
   Standardizes input images for consistent analysis.

2. Preprocessing 
   Enhances contrast, removes noise, and converts images to grayscale for better segmentation.

3. Segmentation 
   Applies morphological filters and the Watershed algorithm to detect and segment tumor regions.

4. Feature Extraction and Classification
   Extracts shape and texture features and classifies MRI images using pre-trained CNN models.

5. Tumor Volume Estimation
   Calculates tumor area and percentage of affected brain tissue for quantitative evaluation.

6. Visualization  
   Generates overlay images to display segmented tumors on the original MRI scans for interpretability.

## Results and Impact

- Enhanced MRI image clarity and contrast.  
- Accurate segmentation and localization of tumor regions.  
- Reliable tumor volume and affected area estimation.  
- Improved interpretability with overlay visualizations.  
- Optimized classification performance through comparative evaluation of multiple CNN models.

## Potential Applications

- Clinical decision support for radiologists.  
- Quantitative tracking of tumor progression.  
- Research in automated medical imaging analysis.  
- AI-assisted oncology diagnostics.
