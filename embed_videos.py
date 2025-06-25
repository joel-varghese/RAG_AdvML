import os

import psycopg2
import torch
import openai
import azure.identity
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector


load_dotenv(override=True)

DBUSER = os.environ["DBUSER"]
DBPASS = os.environ["DBPASS"]
DBHOST = os.environ["DBHOST"]
DBNAME = os.environ["DBNAME"]
# Use SSL if not connecting to localhost
DBSSL = "disable"
if DBHOST != "localhost":
    DBSSL = "require"

conn = psycopg2.connect(database=DBNAME, user=DBUSER, password=DBPASS, host=DBHOST, sslmode=DBSSL)
conn.autocommit = True
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

register_vector(conn)

# Add embedding column if not exists
# cur.execute("ALTER TABLE sports_videos ADD COLUMN IF NOT EXISTS embedding vector(256)")

# cur.execute("CREATE INDEX ON sports_videos USING hnsw (embedding vector_l2_ops)")

# # For each row in the table, compute an embedding using an embedding model
cur.execute("SELECT * FROM sports_videos ORDER BY title DESC")

rows = cur.fetchall()

model_name = "intfloat/e5-small-v2"
device = torch.device("cpu")
model = SentenceTransformer(model_name).to(device)


def get_embedding(text):
    with torch.no_grad():
        embedding = model.encode(text, convert_to_tensor=True, device=device)
    return embedding


for row in rows:
    # if row[3] is not None:
    #     continue
    string_to_embed = row[2] + " " + row[3]
    # Compute the embedding for the string

    embedding = get_embedding(string_to_embed)

    # Update the row with the computed embedding
    cur.execute("UPDATE sports_videos SET embedding = %s WHERE id = %s", (embedding, row[0]))
    print(f"Updated embedding for {row[1]}")






    credential = azure.identity.DefaultAzureCredential()
    token_provider = azure.identity.get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    client = openai.AzureOpenAI(
        api_version="2024-03-01-preview",
        azure_endpoint="https://cog-xw55anu4yrb3k.openai.azure.com",
        azure_ad_token_provider=token_provider,
    )

    response = client.embeddings.create(
        # Azure OpenAI takes the deployment name as the model name
        model="emb3sm",
        input=string_to_embed,
        dimensions=256,
    )
    embedding = response.data[0].embedding
    embedding = np.array(embedding)
    print(embedding)


cur.close()
conn.close()
