from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import spacy
from nltk.stem import PorterStemmer

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "Hello, world! This is a test."

# Step 1 and 2: Tokenization and Stop Word Removal using Gensim
tokens = simple_preprocess(text)
filtered_tokens = [token for token in tokens if token not in STOPWORDS]
print("Filtered Tokens:", filtered_tokens)

# Step 3: Stemming using NLTK
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)

# Step 4: Lemmatization using spaCy
doc = nlp(' '.join(filtered_tokens))
lemmatized_tokens = [token.lemma_ for token in doc]
print("Lemmatized Tokens:", lemmatized_tokens)
