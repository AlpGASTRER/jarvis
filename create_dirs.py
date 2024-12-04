import os

directories = [
    'src',
    'src/voice',
    'src/nlp',
    'src/utils'
]

for dir in directories:
    os.makedirs(dir, exist_ok=True)
