<<<<<<< HEAD
# Fake News Detection using Machine Learning & Deep Learning

This project detects whether a news article is **Fake or Real** using multiple AI models.

The system uses both **Machine Learning (SVM)** and **Deep Learning (DistilBERT)** for prediction and provides an interactive **Streamlit dashboard**.

---

## Dataset

The model is trained on multiple datasets:

- ISOT Fake News Dataset
- WELFake Dataset
- LIAR Dataset
- FakeNewsNet (PolitiFact)

Total dataset size: **~127,000 news articles**

---

## Technologies Used

- Python
- Scikit-learn
- PyTorch
- HuggingFace Transformers
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Models Used

| Model | Accuracy |
|------|----------|
| Naive Bayes | ~86% |
| Logistic Regression | ~93% |
| SVM | **~94%** |
| DistilBERT | **~96-97%** |

---

## Evaluation Metrics

The following evaluation techniques were used:

- Confusion Matrix
- ROC Curve
- Accuracy Score
- Model Comparison Graph

---

## Streamlit Dashboard

The project includes an interactive dashboard where users can:

1. Enter news text
2. Detect whether it is **Fake or Real**
3. View prediction confidence
4. Compare ML vs Deep Learning models

Run the dashboard:

```bash
streamlit run app.py
=======
# Fake_News_Detection
>>>>>>> b2858a6ead87bf3071ab1c1296eeb9d2a4c89dac
