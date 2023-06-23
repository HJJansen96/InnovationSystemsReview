import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_abstract(abstract):
    # Tokenize the abstract
    tokens = word_tokenize(abstract)
    
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation and special characters
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens if token.translate(table) != '']
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

# Read input CSV file
input_file = 'Abstract_CSV.csv'
output_file = 'Abstract_TOKENIZED.csv'

df = pd.read_csv(input_file, encoding='latin-1')

# Preprocess abstracts
df['Preprocessed Abstract'] = df['Abstract'].apply(preprocess_abstract)

# Save preprocessed data to output CSV file
df.to_csv(output_file, index=False)
