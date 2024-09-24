# Health Evaluator

This Python program evaluates health conditions based on user-provided symptoms and optional medical documents. It predicts possible diseases and medications, provides advice based on the severity of symptoms, and suggests nearby medical centers if necessary.Can be merged with a MERN Stack.It provides an integrated solution for medical profession problems.

## Features
- Predicts the top 3 possible diseases based on symptoms.
- Predicts the top 3 medications relevant to the identified diseases.
- Provides medical advice based on the severity of the predicted conditions.
- Identifies nearby medical centers if the condition is severe.
- Optionally performs OCR (Optical Character Recognition) on medical documents to extract and analyze information.

## Installation

1. Clone the repository.
2. Install the required Python packages:

```bash
   pip install geocoder overpy geopy numpy scikit-learn pytesseract pillow nltk
```
OR 
```bash
   pip install -r requirements.txt
```
3.Run using the following format:
```bash
   python HC.py "Your Name" "symptom1;symptom2;symptom3" "path_to_document_image.jpg"
```
FOR EXAMPLE 
## Input:
```
py HC.py "John Doe" "fever;cough;fatigue" 
```
## Output:
```

{
  "name": "John Doe",
  "symptoms": "fever;cough;fatigue",
  "top_3_diseases": [
    "Mixed Incontinence",
    "Gout",
    "Mononucleosis"
  ],
  "top_3_medications": [
    "Stent Placement",
    "Mexiletine",
    "Clopidogrel"
  ],
  "severity": "mild",
  "advice": "Monitor your symptoms and rest.",
  "recommendations": [
    "stay hydrated",
    "rest",
    "take over-the-counter medications if necessary"
  ],
  "location": {
    "latitude": 22.5769,
    "longitude": 88.3186
  }
}
```

