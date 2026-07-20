# 🌊 Rising Waters: Flood Prediction System

## Project Overview

Rising Waters is a Machine Learning-based Flood Prediction System developed as part of the SmartBridge Internship. The application predicts the probability of flooding based on environmental and geographical factors.

## Features

- Predicts flood probability using Machine Learning
- Displays Flood Risk Level (Low, Moderate, High)
- User-friendly web interface using Flask
- Real-time prediction

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML
- CSS

## Dataset

The project uses the Flood Prediction dataset containing 20 input features and 1 target variable (FloodProbability).

## Project Structure

```
RISING_WATERS/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
├── model/
├── static/
└── templates/
```

## How to Run

1. Install the required libraries:

```
pip install -r requirements.txt
```

2. Train the model:

```
python train_model.py
```

3. Run the Flask application:

```
python app.py
```

4. Open your browser and visit:

```
http://127.0.0.1:5000
```

## Author

SmartBridge Internship Project
