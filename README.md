# Smart Retail Shelf Monitoring
## Shoplifting Detection from CCTV Footage

This project aims to detect shoplifting events from CCTV footage using a combination of spatio-temporal deep learning and pose estimation. It fulfills the requirement of going beyond simple classifications by utilizing video data, building an ensemble model, performing data normalization, and deploying a modern Django web interface.

---

## 🏗️ Notebook Pipeline

The core machine learning pipeline is centralized in the `Final_Project.ipynb` notebook. The pipeline consists of the following steps:

1. **Understand & Visualize the Data**: Analyzing the Kaggle Innovatiana Shoplifting Video Dataset.
2. **Normalize the Data**: Every video is re-sampled to 15 FPS by frame-index sampling (without external dependencies like FFmpeg) and every frame is resized to 224x224.
3. **Feature Extraction**: Three independent feature streams are computed and cached:
   - *Pose sequence*: MediaPipe PoseLandmarker (33 keypoints x 4 dimensions).
   - *HOG sequence*: OpenCV HOGDescriptor for edge/gradient silhouette information.
   - *Motion vector*: Dense optical flow magnitude (Farneback).
4. **Data Annotation & Quality Filter**: Filtering out corrupted or empty frames.
5. **Data Split**: Splitting the dataset into 75% training, 15% validation, and 10% testing.
6. **Build Feature Matrices**: Aggregating frame-level features into clip-level statistics.
7. **Model Training**: Training the baseline models and the ensemble (detailed below).
8. **Evaluation**: Generating confusion matrices, performance metrics, and checking for overfitting/underfitting.
9. **Person Counter (Model D)**: A regression model that counts the number of people in a frame.
10. **Shoplifting Frame Locator**: A sliding-window approach that scores individual frames using the ensemble model to find the specific moment of the shoplifting event.
11. **Export**: Exporting the trained models and preprocessing scalers for deployment in the Django backend.

---

## 🧠 Models Used

The project uses a diverse set of models to ensure robust classification. All models are built using `scikit-learn` pipelines (`StandardScaler` -> `Classifier`/`Regressor`).

| Model | Type | Input Features | Output |
|-------|------|----------------|--------|
| **Model A** | RandomForest Classifier | Pose sequence statistics | Shoplifting / Normal |
| **Model B** | GradientBoosting Classifier | HOG frame stats + motion | Shoplifting / Normal |
| **Model C** | Multi-Layer Perceptron (MLP) | Pose sequence statistics | Shoplifting / Normal |
| **Ensemble** | Weighted Soft-Vote | Models A, B, and C | Shoplifting / Normal |
| **Model D** | RandomForest Regressor | Per-frame pose + HOG stats | Person count (integer) |

### Transparent and Inspectable (Model D)
**Model D** is completely transparent: it answers the question *"how many people are in this frame?"* using a RandomForest Regressor. 
- Every decision tree in the forest is fully inspectable.
- Feature importances are plotted so you can see exactly which pose landmarks or HOG features drive the count estimate.
- Ground truth is obtained by running MediaPipe PoseLandmarker with `num_poses=10`.

### Shoplifting Frame Locator
The ensemble (A+B+C) was trained at the *clip* level. To localise the event to a specific frame, we use a sliding-window approach. The frame with the highest shoplifting confidence is flagged as the event frame. This confidence score is simply each classifier's `predict_proba` on a single frame's feature vector, making the temporal localization highly interpretable.

---

## 🗂 Project Structure

The project is organized to ensure clean separation between raw data, processed outputs, model assets, and deployment code. 

```text
FinalProject/
├── data/                       
│   ├── raw/                    <- Place unzipped Kaggle dataset here
│   ├── normalized/             <- Intermediate normalized video files (15 FPS, 224x224)
│   └── processed/              <- 75/15/10 split videos ready for model training
├── models/                     <- Trained weights and best_ensemble_model.joblib
├── django_backend/             <- Web application for CCTV inference dashboard
├── Final_Project.ipynb         <- Main notebook containing all ML pipeline logic
├── requirements.txt            <- Python dependencies
└── README.md                   <- Project documentation
```

## 🚀 Web Application
The Django web application provides a beautiful, modern interface to upload CCTV footage, run the trained ensemble models via OpenCV, and immediately display the localized screenshot of the shoplifting incident with person-tracking overlays.