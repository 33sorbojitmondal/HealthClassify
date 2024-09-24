# Health Evaluator

This Python script evaluates health conditions based on user-provided symptoms and optional medical documents. It predicts possible diseases and medications, provides advice based on the severity of symptoms, and suggests nearby medical centers if necessary.

## Features
- Predicts the top 3 possible diseases based on symptoms.
- Predicts the top 3 medications relevant to the identified diseases.
- Provides medical advice based on the severity of the predicted conditions.
- Identifies nearby medical centers if the condition is severe.
- Optionally performs OCR (Optical Character Recognition) on medical documents to extract and analyze information.

## Installation

1. Clone the repository or download the script.
2. Install the required Python packages:

```bash
   pip install geocoder overpy geopy numpy scikit-learn pytesseract pillow nltk
```
