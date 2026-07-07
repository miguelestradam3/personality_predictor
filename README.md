# Personality Analysis using Machine Learning

An end-to-end data analysis and machine learning project that accurately predicts whether an individual has an **Introvert** or **Extrovert** personality profile based on their social behavior, habits, and preferences. 

This project demonstrates the power of `scikit-learn` for predictive analytics, handling class imbalances, and data visualization in Python.

## 🧑‍💻 Author
* **Miguel Estrada**

## 📊 Dataset Overview
The information was obtained from **Kaggle** (`personality_dataset.csv`). After removing incomplete entries, the dataset consists of **2,477 rows** with a balanced distribution of classes:
* **Extroverts:** 1,274 samples
* **Introverts:** 1,203 samples

### Key Features Analyzed:
* `Time_spent_Alone`: Amount of daily time spent by oneself.
* `Stage_fear`: Presence of stage fright or public speaking anxiety (Yes / No).
* `Social_event_attendance`: Frequency of going to parties or community gatherings.
* `Going_outside`: General tendency to step out of the house.
* `Drained_after_socializing`: Feeling depleted of social energy after interacting with people (Yes / No).
* `Friends_circle_size`: The total number of friends in their close social network.
* `Post_frequency`: Frequency of publishing posts on social media networks.
* `Personality`: Target category (**Introvert** vs. **Extrovert**).

---

## 🚀 Highlights & Data Insights
Before training the model, an exploratory analysis revealed stark differences between behavioral attributes:
* **Circle of Friends:** Extroverts maintain a significantly higher average friend circle size (~9.17) compared to Introverts (~3.20).
* **Going Outside:** Extroverts spend notably more time going outside (mean score of ~4.63) compared to Introverts (mean score of ~1.27).

---

## 🛠️ Project Pipeline

1. **Data Preprocessing & Cleaning:** * Dropped null values to ensure dataset integrity.
   * Converted categorical text columns (`Personality`, `Drained_after_socializing`, `Stage_fear`) into numeric formats using `LabelEncoder`.
2. **Exploratory Data Analysis (EDA):**
   * Generated behavioral comparisons and correlation insights via data visualizations using `seaborn` and `matplotlib`.
3. **Feature Scaling:**
   * Scaled features using `StandardScaler` to normalize the variance across all independent variables.
4. **Handling Imbalances & Oversampling:**
   * Implemented `SMOTE` (Synthetic Minority Over-sampling Technique) via `imblearn` to stabilize training boundaries.
5. **Model Training:**
   * Fitted a robust `LogisticRegression` classification model.

---

## 📈 Model Performance & Evaluation

The Logistic Regression model performs exceptionally well in categorizing individual personality types, yielding an overall accuracy of **~91.45%**.

### Classification Metrics:
```text
               precision    recall  f1-score   support

   Extrovert       0.95      0.89      0.92       335
   Introvert       0.88      0.94      0.91       285

    accuracy                           0.91       620
   macro avg       0.91      0.92      0.91       620
weighted avg       0.92      0.91      0.91       620
