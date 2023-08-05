import joblib

def classify_text(input_text, model_path="text_classifier_model.pkl", vectorizer_path="text_vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    text_vec = vectorizer.transform([input_text])
    
    # Предсказываем вероятности для каждого класса
    probabilities = model.predict_proba(text_vec)[0]
    
    # Получаем класс с максимальной вероятностью и его вероятность
    max_prob_index = probabilities.argmax()
    max_prob = probabilities[max_prob_index]
    predicted_class = model.classes_[max_prob_index]
    
    # Преобразуем вероятность в проценты
    confidence_percentage = max_prob * 100
    
    return predicted_class, confidence_percentage


def get_user_feedback(predicted_status):

    """Получить обратную связь от пользователя."""

    print(f"Предсказанный статус: {predicted_status}")
    feedback = input("Это правильно? (да/нет): ").strip().lower()
    
    if feedback == "нет":
        if predicted_status == "Приемлемый контент":
            return "Неприемлемый контент"
        else:
            return "Приемлемый контент"
    return None



sample_text = "Работу сделал безрукий баклан"
status, confidence = classify_text(sample_text)
print(f"Статус текста '{sample_text}': '{status}' с уверенностью в {confidence:.2f}%")

correct_status = get_user_feedback(status)
if correct_status:
    with open("feedback_data.csv", "a", encoding="utf-8") as file:
        file.write(f'"{sample_text}","{correct_status}"\n')