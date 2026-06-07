from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5.5",
    # reasoning={"effort": "high"},
    input=
    """
Say hi

"""
)

print(response.output_text)
