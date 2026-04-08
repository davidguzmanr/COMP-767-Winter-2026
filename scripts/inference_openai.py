import base64
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL_ID = "gpt-4o-mini"
IMAGE_PATH = "images/Dia-de-Muertos-1.jpg"
COUNTRY = "Mexico"
QUESTION = "What is shown in this image?"

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "system",
            "content": (
                f"You are a helpful assistant with expertise in {COUNTRY} culture. "
                f"When answering questions, consider the cultural context of {COUNTRY}."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": QUESTION},
            ],
        },
    ],
    logprobs=True,
    top_logprobs=5,
    max_completion_tokens=64,
    temperature=0.7,
)

choice = response.choices[0]
print(f"Response: {choice.message.content}\n")

# print(response)

for i, token_lp in enumerate(choice.logprobs.content):
    print(f"Token {i}: {token_lp.token!r}  logprob={token_lp.logprob:.4f}")
    for alt in token_lp.top_logprobs[:3]:
        print(f"  alt: {alt.token!r}  logprob={alt.logprob:.4f}")