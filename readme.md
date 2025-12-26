# 📑 Big Data Challenge (BDC) Satria Data 2025 – Emotion Classification

## 📌 Introduction

The **Big Data Challenge (BDC) Satria Data 2025** competition challenges participants to build an emotion classification model based on multimedia data (video).
Each video must be classified into one of the following 8 emotion categories:

* Proud
* Trust
* Joy
* Surprise
* Neutral
* Sadness
* Fear
* Anger

The primary evaluation metric is **Macro-averaged F1-Score**.

<div align="center">
  <img src="src/t-SNE.png" alt="t-SNE Visualization" width="600">
</div>

---

## 📂 Dataset Structure

The organizers provided two CSV files:

* **datatrain.csv** → `id, video (IG URL), emotion` (803 labeled videos)
* **datatest.csv** → `id, video (IG URL)` (200 unlabeled videos, for submission)

Observation results:

* Train: 803 total → ✅ 775 successfully downloaded, ❌ 28 failed (videos deleted/private)
* Test: 200 total → ✅ 198 successfully downloaded, ❌ 2 failed

Videos were downloaded using `yt_dlp` and stored in the following folder structure:

```
data/
  raw/              # original CSV files from the organizers
  processed/        # corrected CSV files (typo labels fixed)
  video/
    train/
    test/
```

---

## ⚙️ Preprocessing & Feature Extraction

1. **Label Correction**

   * Several label typos (e.g., `trst` → `trust`) → cleaned and saved in `processed/`.

2. **Audio Extraction**

   * Convert `.mp4 → .wav` using `ffmpeg`
   * Feature extraction using **Librosa** → saved as `.npy`
   * Structure: `features/audio/{train,test}`

3. **Visual Extraction**

   * Extract frames per video using `cv2`
   * Frame-level feature extraction using **ResNet50** (2048 dimensions)
   * Saved in `features/visual/`

4. **Text Extraction**

   * Speech transcription using **Whisper** → text
   * Text representation using **IndoBERT** (768 dimensions)
   * Saved in `features/text/`

5. **Feature Fusion**

   * Audio (20) + Visual (2048) + Text (768) → Total **2836 dimensions** per video
   * Final dataset shape: `(802, 2836)`

---

## 🔍 Exploratory Data Analysis (EDA)

* **Imbalanced label distribution** → *Surprise* is dominant, *Neutral* is the least frequent.
* **t-SNE visualization** → clusters are not clearly separated, classes are mixed.
* **Label EDA**

  * Number of unique classes: 8
  * Most frequent label: *Surprise*
  * Least frequent label: *Neutral*

---

## 🧠 Modeling

### Approach:

* Main model: **XGBoost Classifier**
* Imbalance handling: **SMOTE**
* Validation: **5-fold Cross-validation**
* Additional evaluation: **Macro ROC-AUC**

### Evaluation Results

* **Confusion Matrix (summary):**

  * Anger, Fear, Joy, Neutral, Sadness → nearly perfect
  * Proud, Surprise, Trust → frequently confused (semantically similar)

* **Metrics:**

  * Macro ROC-AUC: **0.9844**
  * Cross-val Macro F1 (per fold): `[0.8803, 0.8829, 0.8767, 0.8556, 0.8838]`
  * Mean CV Macro F1: **0.8759**

👉 Conclusion:
The model shows **very strong performance**, with the main challenge being the **Proud / Surprise / Trust** classes, which are emotionally close.

---

## 📊 Conclusion

* An **end-to-end pipeline** was successfully built, from raw video → multimodal features → classification model.
* Evaluation results show high performance (**Macro F1 ≈ 0.876**) and consistency across folds.
* The main challenge lies in the overlap between similar emotions (*Proud, Surprise, Trust*).

---

## 🚀 Next Steps

* **Optional improvements:**

  * Hyperparameter tuning (XGBoost, Random Forest, LightGBM)
  * Experiments with more advanced multimodal models (e.g., fusion layers, transformer-based multimodal models)
  * Additional feature engineering

* **For Submission:**

  * Test predictions are saved in `submission+TeamName.csv` format following the provided template.
  * Column format: `id, predicted` (0–7 according to label mapping).

---

## 📦 Reproducibility

1. **Environment**

   * Recommended to use **Conda/venv**
   * Create `requirements.txt` using `pip freeze > requirements.txt`
   * Alternatively, create `environment.yml` for conda

2. **Final Folder Structure:**

```
BDC2025/
  data/
    processed/
    raw/
    video/
    wav/
  features/
  notebooks/
    OO_checking_emotion.ipynb
    OO_estimation.ipynb
    Ol_download_videos.ipynb
    02_feature_extraction.ipynb
    03_feature_eda.ipynb
    04_modeling.ipynb
    05_submission.ipynb
  src/
  submission/
  README.md
  requirements.txt
```

---

## 📝 Notes

* Some videos failed to download (28 train, 2 test) → left as missing.
* Librosa experienced crashes → resolved by downgrading the version.
* Feature extraction required a considerable amount of time (~5 hours total).
