'''Flask - создание веб-приложений
render_tamlate - модуль для отрисовки html-шаблонов
request - получение данных из POST/GET запросов
Transformers - готовый пайплан для ML-задач
pipline - готовый паплайн
AutoTokenizer - токенизатор для преобразования текста в числа
AutoModelForCauselLM - модель для генерации текста
re - модуль для регулярных выражений
warnings - подавлять предупреждений
'''
from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLMimport requestimport warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment"
)

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")


#Функция получения названия фильма
def extract_film_title(generated_text: str) -> str:
    if not generated_text:
        return "Не удалось получить рекомендацию!"

        text = generated_text.strip()
        m = re.search(r'<<([^>>]{2,100})>>', text)
        if m:
            return m.group(1).strip()
        
        first_line = text.split('\n')[0].strip()
        first_line = re.sub(r'[.,:;!?]+$', '', first_line).strip()

        if first_line and len(first_line) > 3:
            return first_line[:100]
        
        return "Не удалось получить рекомендацию!"
    
    # Функция генерация рекомендации
    def generate_recommendation(mood):
        promt = (
            f"Посоветуй только один популярный фильм для человека, у которого {mood} настроение."
            f"Напиши только название фильма, без описания и коментариев."
        )

        inputs = tokenizer(prompt, return_tensor="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True, # Включение случайности
            top_p=0.7,
            temperature=0.5,
            num_return_sequences=1
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(promt):].strip()
    
    @app.route("/", methods=["GET", "POST"])
    def index():
        recommendation=""
        user_text=""

        if request.methood == "POST":
            user_text = request.form["message"]
            result = sentiment_analyzer(user_text)[0]
            label = result['label']

            if label == "POSITIVE":
                mood = "хорошее"
            elif label == "NEGATIVE":
                mood = "плохое"
            else:
                mood = "нетральное"
            ai_text = extract_film_title(generate_recommendation(mood))
            recommendation = f"Ваше настроение: {mood}. Рекомендация фильма: {ai_text}"
            return render_template("index.html", recommendation=recommendation, user_text=user_text)
        
        if __name__== "__main__":
            app.run(debug=True)

