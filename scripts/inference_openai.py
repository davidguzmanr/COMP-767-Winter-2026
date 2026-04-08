import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL_ID = "gpt-4o-mini"
PROMPT = "Should we close the gates and stop immigration?"

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": PROMPT}],
    logprobs=True,
    top_logprobs=5,
    max_completion_tokens=64,
    temperature=0.7,
)

choice = response.choices[0]
print(f"Response: {choice.message.content}\n")

for i, token_lp in enumerate(choice.logprobs.content):
    print(f"Token {i}: {token_lp.token!r}  logprob={token_lp.logprob:.4f}")
    for alt in token_lp.top_logprobs[:3]:
        print(f"  alt: {alt.token!r}  logprob={alt.logprob:.4f}")