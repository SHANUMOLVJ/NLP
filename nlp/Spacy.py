import spacy
from nltk.stem import PorterStemmer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "Hello, world! This is a test."

# Step 1: Tokenization and Step 2: Stop Word Removal (using spaCy)
doc = nlp(text)
filtered_tokens = [token for token in doc if not token.is_stop and not token.is_punct]
filtered_token_texts = [token.text for token in filtered_tokens]
print("Filtered Tokens:", filtered_token_texts)

# Step 3: Stemming (using NLTK)
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_token_texts]
print("Stemmed Tokens:", stemmed_tokens)

# Step 4: Lemmatization (using spaCy)
lemmatized_tokens = [token.lemma_ for token in filtered_tokens]
print("Lemmatized Tokens:", lemmatized_tokens)
