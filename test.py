
import os
from openai import AzureOpenAI


os.environ['AZURE_OPENAI_API_KEY'] = 'ba80dab708f34541894c844dd29071b3'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://openai-gtp4-baixing.openai.azure.com/'
'''
os.environ['AZURE_OPENAI_API_KEY'] = '9cd7d887a86a4f34932bd8f2231b1522'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://bxaisc.openai.azure.com/'
'''

client = AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version="2024-05-13",
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )

deployment_name = 'gpt-4o'

print('Sending a test completion job')
start_phrase = 'Write a tagline for an ice cream shop. '
response = client.completions.create(model=deployment_name, prompt=start_phrase, max_tokens=10)
print(start_phrase+response.choices[0].text)