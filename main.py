from together import Together
from resp_evaluator import Evaluator
import json
from keys import api_key

user_data = ['name', 'age', 'gender']
model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

client = Together(api_key=api_key)
main_evaluator = Evaluator(api_client=client, api_key=api_key)


messages_chatbot = [
    {"role": "system", "content": f"Talk to the user while interviewing them for: {', '.join(user_data)}."}
]

while True:
    user_message = input("user: ")
    messages_chatbot.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages_chatbot
    )

    text_response = response.choices[0].message.content
    messages_chatbot.append({"role": "assistant", "content": text_response})

    # Send messages to evaluator to extract structured data
    main_evaluator.submit_message({"role": "user", "content": "target of interview: " + user_message})
    main_evaluator.submit_message({"role": "user", "content": "interviwer: " + text_response})

    new_data = main_evaluator.evaluate()
    print(f"Collected data so far: {new_data}")
    print(f"Assistant response: {text_response}")
