import csv
import geocoder
import overpy
from geopy.distance import geodesic
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import json
import pytesseract
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def load_health_data(csv_file):
    health_data = []
    try:
        with open(csv_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if all(row.values()):
                    health_data.append({
                        'symptoms': row['symptoms'].split(';'),
                        'disease': row['disease'],
                        'medication': row['medication']
                    })
        if not health_data:
            raise ValueError("No valid data found in the CSV file.")
        return health_data
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    except csv.Error as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

def preprocess_data(health_data):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform([item['symptoms'] for item in health_data])
    
    disease_encoder = LabelEncoder()
    medication_encoder = LabelEncoder()
    
    y_disease = disease_encoder.fit_transform([item['disease'] for item in health_data])
    y_medication = medication_encoder.fit_transform([item['medication'] for item in health_data])
    
    return X, y_disease, y_medication, mlb, disease_encoder, medication_encoder

def train_models(X, y_disease, y_medication):
    X_train, X_test, y_disease_train, y_disease_test, y_medication_train, y_medication_test = train_test_split(
        X, y_disease, y_medication, test_size=0.2, random_state=42)

    disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
    medication_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    disease_model.fit(X_train, y_disease_train)
    medication_model.fit(X_train, y_medication_train)
    
    return disease_model, medication_model

def predict_health_info(symptoms, mlb, disease_encoder, medication_encoder, disease_model, medication_model):
    X = mlb.transform([symptoms.split(';')])
    
    disease_proba = disease_model.predict_proba(X)[0]
    medication_proba = medication_model.predict_proba(X)[0]
    
    top_3_diseases = disease_encoder.inverse_transform(disease_proba.argsort()[-3:][::-1])
    top_3_medications = medication_encoder.inverse_transform(medication_proba.argsort()[-3:][::-1])
    
    return top_3_diseases, top_3_medications

def get_real_time_location():
    try:
        g = geocoder.ip('me')
        return g.latlng if g.latlng else (0, 0)
    except Exception as e:
        print(f"Error getting location: {str(e)}")
        return (0, 0)

def get_nearby_medical_centers(latitude, longitude):
    try:
        api = overpy.Overpass()
        query = f"""
        [out:json];
        (
          node["amenity"="hospital"](around:5000,{latitude},{longitude});
          node["amenity"="clinic"](around:5000,{latitude},{longitude});
        );
        out body;
        """
        result = api.query(query)
        medical_centers = []
        for node in result.nodes:
            name = node.tags.get("name", "Unknown")
            lat = node.lat
            lon = node.lon
            distance = round(geodesic((latitude, longitude), (lat, lon)).kilometers, 2)
            phone = node.tags.get("phone", "N/A")
            medical_centers.append({
                'name': name,
                'distance': distance,
                'phone': phone
            })
        medical_centers.sort(key=lambda x: x['distance'])
        return medical_centers[:5]
    except Exception as e:
        print(f"Error fetching medical centers: {str(e)}")
        return []

def get_medical_advice(predicted_diseases, predicted_medications, latitude, longitude):
    severity = np.random.choice(['mild', 'moderate', 'severe'], p=[0.6, 0.3, 0.1])
    
    if severity == 'severe':
        advice = "Seek immediate medical attention."
        medical_centers = get_nearby_medical_centers(latitude, longitude)
        return severity, advice, medical_centers
    elif severity == 'moderate':
        advice = "Consult with a healthcare provider soon."
        return severity, advice, predicted_medications
    else:
        advice = "Monitor your symptoms and rest."
        home_remedies = ["stay hydrated", "rest", "take over-the-counter medications if necessary"]
        return severity, advice, home_remedies

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def extract_medical_info(text):
    tokens = preprocess_text(text)
    diseases = [token for token in tokens if token in ['diabetes', 'hypertension', 'asthma']]
    medications = [token for token in tokens if token in ['insulin', 'metformin', 'lisinopril']]
    return diseases, medications

def ocr_medical_document(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        return ""

def analyze_medical_document(image_path):
    text = ocr_medical_document(image_path)
    if text:
        diseases, medications = extract_medical_info(text)
        return {
            "extracted_text": text,
            "identified_diseases": diseases,
            "identified_medications": medications
        }
    else:
        return {"error": "Failed to extract text from the image"}

def evaluate_health(name, symptoms, medical_document_path=None):
    csv_file = 'processed_health_data.csv'
    try:
        health_data = load_health_data(csv_file)
        X, y_disease, y_medication, mlb, disease_encoder, medication_encoder = preprocess_data(health_data)
        disease_model, medication_model = train_models(X, y_disease, y_medication)
        
        top_3_diseases, top_3_medications = predict_health_info(symptoms, mlb, disease_encoder, medication_encoder, disease_model, medication_model)
        
        latitude, longitude = get_real_time_location()
        
        severity, advice, recommendations = get_medical_advice(top_3_diseases, top_3_medications, latitude, longitude)
        
        result = {
            "name": name,
            "symptoms": symptoms,
            "top_3_diseases": list(top_3_diseases),
            "top_3_medications": list(top_3_medications),
            "severity": severity,
            "advice": advice,
            "recommendations": recommendations,
            "location": {"latitude": latitude, "longitude": longitude}
        }
        
        if medical_document_path:
            document_analysis = analyze_medical_document(medical_document_path)
            result["medical_document_analysis"] = document_analysis
        
        return result
    
    except Exception as e:
        return {"error": str(e)}

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Invalid arguments"}))
        sys.exit(1)

    name = sys.argv[1]
    symptoms = sys.argv[2]
    medical_document_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = evaluate_health(name, symptoms, medical_document_path)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()