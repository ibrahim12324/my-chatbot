from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load your business information from a text file
def load_business_info(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Initialize the question-answering pipeline
def create_qa_pipeline():
    return pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

# Get an answer from the model
def get_answer(question, context, qa_pipeline):
    result = qa_pipeline(question=question, context=context)
    if result['score'] < 0.2:
        return "I'm sorry, I don't have an answer to that question. Could you please provide more details or rephrase your question?"
    return result['answer']

# Handle small talk and specific queries
def handle_small_talk(question):
    small_talk = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! What can I do for you?",
        "what's your name": "I am the Tech Solutions chatbot. How can I help you?",
        "bye": "Goodbye! Have a great day!",
        "exit": "Goodbye! Have a great day!",
        "quit": "Goodbye! Have a great day!",
        "how are you": "I'm just a bot, but I'm here to help you! How can I assist you today?",
        "what do you do": "We specialize in developing AI chatbots for businesses. How can our services help you today?",
        "when were you founded": "We were founded in August 2024. Though we are new, we are excited to bring innovative chatbot solutions to the market.",
        "what do you sell": "We offer AI-powered chatbot solutions, including custom chatbot development, integration with various platforms, consulting and strategy, ongoing support and maintenance, and training workshops."
    }
    return small_talk.get(question.lower(), None)

# Load the business information
context = load_business_info('business_info.txt')

# Create the QA pipeline
qa_pipeline = create_qa_pipeline()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('question')
    small_talk_response = handle_small_talk(user_input)
    if small_talk_response:
        response = {"response": small_talk_response}
        if user_input.lower() in ['exit', 'quit']:
            response["exit"] = True
        return jsonify(response)
    else:
        answer = get_answer(user_input, context, qa_pipeline)
        return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)

