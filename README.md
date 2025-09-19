# Cardiac Pathology Prediction

Python project developed for the **IM05 Challenge (Télécom Paris, 2025)**.  
The project involves predicting cardiac conditions from CMRI images and performing left ventricle segmentation.

---

## Project Overview

This project includes two main tasks:

1. **Cardiac Condition Classification**  
   - Predicts cardiac conditions from CMRI images into five diagnostic classes.  
   - Implements a Random Forest classifier with feature selection.  
   - Dataset and evaluation follow a Kaggle-style challenge format.  

2. **Left Ventricle Segmentation**  
   - Segments the left ventricle from CMRI images.  
   - Provides a preprocessing and pipeline setup suitable for medical image segmentation.  

The **main notebook** that integrates all steps of the challenge is CHALLENGE_LUCIANAMUNHOS.ipynb
Final model achieved 88% accuracy on the test set.

---

## Technologies Used

- Language: Python 3  
- Libraries: scikit-learn, numpy, pandas, matplotlib, OpenCV, SimpleITK  
- Machine Learning: Random Forest, feature selection  
- Image Processing: CMRI image preprocessing, segmentation pipeline  
