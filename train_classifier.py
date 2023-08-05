import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import sys


if len(sys.argv) < 2:
    print("Пожалуйста, укажите путь к CSV-файлу.")
    sys.exit()

data_path = sys.argv[1]
data = pd.read_csv(data_path, encoding='utf-8')

# Попробовать загрузить обратную связь от пользователей
try:
    feedback_data = pd.read_csv("feedback_data.csv", encoding='utf-8')
    data = pd.concat([data, feedback_data], ignore_index=True)
except FileNotFoundError:
    pass

# 2. Предобработка данных
X_train, X_test, y_train, y_test = train_test_split(data['Текст'], data['Статус'], test_size=0.1, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. Обучение модели
model = LogisticRegression(random_state=42)
model.fit(X_train_vec, y_train)

# Опционально: оценка модели на тестовых данных
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 4. Сохранение модели и векторизатора
model_path = "text_classifier_model.pkl"
vectorizer_path = "text_vectorizer.pkl"
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)