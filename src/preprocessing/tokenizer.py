import re
import string

import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

STOPWORDS = set(stopwords.words('english'))
STEMMER = SnowballStemmer(language='english')

EMOJI_PATTERN = re.compile(
    '['
    '\U0001F600-\U0001F64F'
    '\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF'
    '\U0001F1E0-\U0001F1FF'
    '\U00002700-\U000027BF'
    '\U000024C2-\U0001F251'
    ']+',
    flags=re.UNICODE
)

def tokenize(text: str) -> list[str]:

    if not text:
        return []
    
    # 1. Lowercase & Remove whitespace from end
    normalized_text = text.lower().strip()

    # 2. Remove emoji & unicode symbol
    normalized_text = EMOJI_PATTERN.sub('', normalized_text)

    # 3. Remove punctuation
    normalized_text = normalized_text.translate(str.maketrans('', '', string.punctuation))

    # 4. Split into tokens
    tokens = normalized_text.split()

    # 5. Remove stopwords & stemming
    tokens = [STEMMER.stem(token)for token in tokens if token not in STOPWORDS]

    return tokens

if __name__ == '__main__':
    text = 'Hello World! This is A Demo TEXT.. 🚀 😊'
    print(tokenize(text))