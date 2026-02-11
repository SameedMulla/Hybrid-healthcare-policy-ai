import ollama

response = ollama.chat(
    model='mistral',
    messages=[
        {"role": "user", "content": "How should Maharashtra deploy vaccines in rural districts?"}
    ]
)

print(response['message']['content'])
