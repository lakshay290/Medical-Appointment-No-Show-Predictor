---

# 🏥 Medical Appointment No-Show Predictor

A **Python machine learning project** with a **Tkinter GUI** that predicts whether a patient will **show up** or **miss** their medical appointment based on demographic, health, and scheduling data.

---

## 🌟 Features

* 📂 **Data Preprocessing** – Cleans and prepares the Kaggle dataset.
* 📊 **Feature Engineering** – Calculates lead time, encodes gender, and handles handicap.
* 🤖 **Machine Learning Model** – Trains a Logistic Regression classifier with scikit-learn.
* 🎨 **GUI with Tkinter** – User-friendly interface to:

  * Enter patient details manually.
  * Predict "Show Up" or "No-Show" with probability.
  * Display random 20-row samples from dataset.
  * Full-screen mode with an exit option.

---

## 📂 Project Structure

```
NoShowPredictor/
│── KaggleV2-May-2016.csv   # Dataset (Kaggle medical appointment dataset)
│── predictor.py            # Main script (data prep, training, GUI)
│── README.md               # Documentation
```

---

## 🛠️ Technologies Used

* **Python 3.9+**
* **Pandas** – Data cleaning & manipulation.
* **Scikit-learn** – Logistic Regression model.
* **Tkinter** – GUI for user interaction.

---

## 🚀 How to Run

1. Clone or download the repository.

   ```bash
   git clone https://github.com/your-username/NoShowPredictor.git
   cd NoShowPredictor
   ```

2. Install dependencies:

   ```bash
   pip install pandas scikit-learn
   ```

3. Download the dataset from Kaggle:
   [Medical Appointment No Shows (Kaggle)](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
   Save it as `KaggleV2-May-2016.csv` in your project folder.

4. Run the predictor:

   ```bash
   python predictor.py
   ```

---

## 🎮 How to Use the App

* Start the app → It opens in **full-screen mode**.
* Enter patient details in the input fields:

  * Age, Scholarship, Hypertension, Diabetes, Alcoholism
  * SMS Received, Lead Time, Handicap, Gender
* Click **Predict** → Get prediction (`Show Up` or `No-Show`) with probability.
* Click **Show Random Data** → See 20 random patient records from dataset.
* Press **Escape (Esc)** → Exit full-screen mode.

---

## 📊 Example Prediction

**Input:**

* Age: 45
* Scholarship: 0
* Hypertension: 1
* Diabetes: 0
* Alcoholism: 0
* SMS Received: 1
* Lead Time: 5
* Handicap: 0
* Gender: F

**Output:**

```
PREDICTION: NO-SHOW
Probability of missing appointment: 67.35%
```

---

## 📌 Future Enhancements

* Add more ML models (Random Forest, XGBoost).
* Display model accuracy on test set in the GUI.
* Save predictions into a local CSV.
* Deploy as a web app with **Streamlit** or **Flask**.

---

## 📜 License

This project is licensed under the **MIT License**.

---
