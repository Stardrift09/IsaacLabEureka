from openai import OpenAI

client = OpenAI()

usage = client.responses.list()  # or usage-related endpoints depending on account access
print(usage)