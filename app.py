from flask import Flask, request, jsonify
from model.ChatbotModel import ChatbotModel

app = Flask(__name__)

# Instantiate chatbot model
chatbot = ChatbotModel(model_path="./model/chatbot")

@app.route('/api/execute_chatbot', methods=['POST'])
def execute_chatbot():
    """
    API endpoint for interacting with the chatbot.

    Request JSON format:
    {
        "question": "What is the meaning of life?",
    }

    Response JSON format:
    {
        "answer": "The answer to the meaning of life is..."
    }
    """
    try:
        data = request.get_json()

        question = data['question']

        # Generate answer using the chatbot
        answer = chatbot.generate_answer(question, 120)

        # Prepare the response
        response = {'answer': answer}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
