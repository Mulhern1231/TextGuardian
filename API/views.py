from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import TextEntry
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

PASSWORD = "1111"
MODEL_PATH = "text_classifier_model.pkl"
VECTORIZER_PATH = "text_vectorizer.pkl"

@csrf_exempt
def train_model(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)

    # Проверка пароля
    password = request.POST.get('password')
    if not password or password != PASSWORD:
        return JsonResponse({'error': 'Invalid password.'}, status=403)

    # Проверка наличия файла
    csv_file = request.FILES.get('csv_file')
    if not csv_file:
        return JsonResponse({'error': 'No CSV file provided.'}, status=400)

    # Загрузка и проверка данных из CSV
    try:
        data = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
        if 'Текст' not in data.columns or 'Статус' not in data.columns:
            return JsonResponse({'error': 'Invalid CSV format.'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Error reading CSV: {str(e)}'}, status=500)

    # Сохранение данных в базу данных
    for _, row in data.iterrows():
        TextEntry.objects.create(text=row['Текст'], status=row['Статус'])

    # Обучение модели
    X = data['Текст']
    y = data['Статус']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vec, y_train)

    # Сохранение модели и векторизатора
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return JsonResponse({'message': 'Model trained successfully.'})


def classify_text(input_text):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    text_vec = vectorizer.transform([input_text])
    probabilities = model.predict_proba(text_vec)[0]
    
    max_prob_index = probabilities.argmax()
    max_prob = probabilities[max_prob_index]
    predicted_class = model.classes_[max_prob_index]
    
    confidence_percentage = max_prob * 100
    
    return predicted_class, confidence_percentage

def get_text_status(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Only GET requests are allowed.'}, status=405)

    text = request.GET.get('text')
    if not text:
        return JsonResponse({'error': 'No text provided.'}, status=400)

    predicted_class, confidence = classify_text(text)
    return JsonResponse({'status': predicted_class, 'confidence': confidence})
