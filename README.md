# Medical Specialty Classification using Bio_ClinicalBERT

This project demonstrates how to build a machine learning model to classify medical documents by specialty using BERT embeddings from Bio_ClinicalBERT. The model can help automatically categorize clinical notes, transcriptions, and other medical text documents based on their specialty area.

## Project Overview

Medical specialty classification has numerous applications, including:
- Automated routing of clinical documents
- Organization and retrieval of medical records
- Trend analysis across medical specialties
- Quality improvement in documentation

This notebook implements a complete workflow using Bio_ClinicalBERT, a domain-specific language model trained on clinical text, to generate embeddings for classification.

## Dataset

The analysis uses the MTSamples dataset, which contains transcribed medical documents across 40 different specialties. After preprocessing and filtering:
- Original dataset: ~5,000 documents across 40 medical specialties
- Filtered dataset: 2,324 documents across 12 medical specialties

## How to Run

This project can be easily run in Google Colab without requiring local installation. Follow these steps to set up your environment and run the code:

### 1. Dataset and Optional Pre-computed Files

You have two options for running this project:

#### Option A: Download Only the Dataset (Run Everything from Scratch)
- Download just the MTSamples dataset directly:
  - [mtsamples.csv](https://raw.githubusercontent.com/neuml/paperai/master/data/mtsamples.csv)
- This option requires more computation time as you'll need to generate embeddings and train models

#### Option B: Download Pre-computed Files (Faster Execution)
- For faster execution, you can download pre-computed embeddings and trained models:
  [Download Pre-computed Files](https://drive.google.com/drive/folders/1-ZTGDCyjoyH2T-lzuZGOiLnkrRVlquMh?usp=drive_link)
- **Important:** After downloading, upload these files to your own Google Drive in a folder called `medical_nlp_models_v3`

The drive folder contains:
- Notebook: `Medical_Specialty_Classification.ipynb`
- Dataset: `mtsamples.csv`
- Saved models: Random Forest, Logistic Regression, and PCA transformer
- Pre-computed embeddings (saves significant computation time)

### 2. Access Google Colab

- Go to [Google Colab](https://colab.research.google.com/)
- Sign in with your Google account

### 3. Upload the Notebook

- Click on "File" → "Upload notebook"
- Select the downloaded `Medical_Specialty_Classification.ipynb` file

### 4. Mount Google Drive and Set Up Environment

- Run the following code to mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Create directory for saved models
import os
save_path = '/content/drive/MyDrive/medical_nlp_models_v3'
if not os.path.exists(save_path):
    os.makedirs(save_path)
```

### 5. Install Required Packages

- Run the following to install necessary libraries:
```python
!pip install transformers torch nltk scikit-learn imbalanced-learn

# Download NLTK resources
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('popular')
```

### 6. Upload Dataset and Models

If you're using the pre-computed files (Option B above):

1. Create a folder in your Google Drive called `medical_nlp_models_v3`
2. Upload all the downloaded files to this folder
3. Ensure the notebook looks for files in the correct path:
   ```python
   save_path = '/content/drive/MyDrive/medical_nlp_models_v3'  # or whichever folder you created
   ```

If you're starting from scratch (Option A):

1. Upload only the `mtsamples.csv` file to your Colab runtime or to your Google Drive
2. The notebook will generate all necessary embeddings and models (this will take longer)
3. These generated files will be saved to your specified `save_path`

### 7. Execute Notebook Cells

- Run each cell in order by clicking the play button or using Shift+Enter
- Note that cells for generating embeddings will use the pre-computed embeddings if available in your Drive path
- If embedding files aren't found, the notebook will generate them (may take 10-15 minutes)

## Key Processing Stages

The notebook contains these main sections:

1. **Data Preprocessing** (Cells 1-11)
   - Loading and cleaning the dataset
   - Filtering specialties with sufficient samples
   - Text normalization using lemmatization

2. **Feature Extraction** (Cells 12-14)
   - Extracting embeddings using Bio_ClinicalBERT
   - This is the most computationally intensive part (pre-computed embeddings save time)

3. **Dimensionality Reduction and Class Balancing** (Cells 15-16)
   - Applying PCA to reduce embedding dimensions
   - Using SMOTE to balance class distributions

4. **Model Training and Evaluation** (Cells 17-22)
   - Training Random Forest and Logistic Regression models
   - Evaluating model performance

5. **Analysis of Excluded Categories** (Cells 23-30)
   - Applying trained models to categories initially excluded
   - Analyzing prediction distributions

## Results

The best performing model was a Random Forest classifier with 5000 trees, achieving:
- Overall accuracy: 74.21%
- Strong performance on specialties with distinct vocabulary and structured notes

Check the full report within the notebook for detailed analysis by specialty.

## Troubleshooting

If you encounter errors related to missing files:
- Ensure that you've uploaded all files from the Google Drive link
- Verify that the file paths in the notebook match your Drive structure
- For embedding generation errors, ensure you have enough Colab runtime memory (consider using a GPU runtime)

## Conclusion

This project demonstrates the effectiveness of domain-specific language models for medical document classification, with potential real-world applications in healthcare document management systems.