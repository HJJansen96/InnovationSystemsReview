import openai
import json
import pandas as pd
# from openai import retry, stop, wait_random_exponential

# Configure your OpenAI API key
openai.api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# Get embeddings
# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

# Read the CSV file
df = pd.read_csv("Abstract_TOKENIZED.csv")

# Get embeddings
embeddings = []
for abstract in df["Preprocessed Abstract"]:
    embedding = get_embedding(json.dumps(abstract), model="text-embedding-ada-002")
    embeddings.append(embedding)

# Convert embeddings to DataFrame
embeddings_df = pd.DataFrame(embeddings)

# Save embeddings to "Abstract_CLUSTERED - test.csv"
embeddings_df.to_csv("Abstract_EMBEDDED.csv", index=False)