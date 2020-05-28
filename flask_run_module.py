from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify
from Stream_img_model import get_predict

"""Запуск веб-сервиса на Flask"""

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run


@app.route('/predict', methods=['GET', 'POST'])
def predict_():
    try:
        # Получение ответа от модели
        json_input = request.json

        # Формирование данных для запроса на сервис
        StreamLink = json_input['StreamLink']
        UserId = json_input['UserId']
        UserMail = json_input['UserMail']


        # перевод числового предсказания в текстовый
        pred = get_predict()

        # Формат ответа
        output = {
            'StreamLink': StreamLink,
            'UserId': UserId,
            'UserMail': UserMail,
            'Found boxes': pred
        }
        return jsonify(output)

    except:

        return "Something Goes Wrong"


if __name__ == '__main__':
    app.run()