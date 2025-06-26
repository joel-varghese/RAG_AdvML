"""
Simple RAG

1. set up postgres connection
2. get question from user
3. Use question to search postgres table
4. Format the results in a LLM-friendly way
5. Send results to the LLM

"""

import os
import psycopg2
import numpy as np
import torch
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


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
register_vector(conn)
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

# get question from user
question = "show me some of the best soccer games streaming now"


model_name = "intfloat/e5-small-v2"
device = torch.device("cpu")
model = SentenceTransformer(model_name).to(device)


def get_stuff(text):
    with torch.no_grad():
        embedding = model.encode(text, convert_to_tensor=True, device=device)
        embedding = embedding.cpu().numpy().tolist()
    return embedding


# cur.execute(
#     "SELECT * FROM sports_videos WHERE title LIKE %s OR description LIKE %s", (f"%{question}%", f"%{question}%")
# )

# results = cur.fetchall()
# print(results)

# Use question to serch Postgres table using buil-in full text search to_tsvector

# cur.execute(
#     "SELECT * FROM sports_videos WHERE to_tsvector(title || ' ' || description) @@ to_tsquery(%s) LIMIT 10", (question,)
# )

# results = cur.fetchall()
# for result in results:
#     print(result[2])

# cur.execute(
#     """
# SELECT id, title, description
#             FROM sports_videos, plainto_tsquery('english', %(query)s) query
#             WHERE to_tsvector('english', description) @@ query
#             ORDER BY ts_rank_cd(to_tsvector('english', description), query) DESC
#             LIMIT 10
# """,
#     {"query": question},
# )

# results = cur.fetchall()
# for result in results:
#     print(result[1])

# Do a Postgres vector embedding search on embedding column with cosine operator

embedding = get_stuff(question)
embedding = np.array(embedding)

# cur.execute("SELECT id, title, description FROM sports_videos ORDER BY embedding <-> %s LIMIT 10", (embedding,))
# results = cur.fetchall()
# for result in results:
#     print(result[2])

cur.execute(
    """
WITH semantic_search AS (
    SELECT id, RANK () OVER (ORDER BY embedding <=> %(embedding)s) AS rank
    FROM sports_videos
    ORDER BY embedding <=> %(embedding)s
    LIMIT 20
),
keyword_search AS (
    SELECT id, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', title || ' ' || description), query) DESC)
    FROM sports_videos, plainto_tsquery('english', %(query)s) query
    WHERE to_tsvector('english', title || ' ' || description) @@ query
    ORDER BY ts_rank_cd(to_tsvector('english', title || ' ' || description), query) DESC
    LIMIT 20
)
SELECT
    COALESCE(semantic_search.id, keyword_search.id) AS id,
    COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
    COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score
FROM semantic_search
FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
ORDER BY score DESC
LIMIT 5
""",
    {"query": question, "embedding": embedding, "k": 60},
)

results = cur.fetchall()

ids = [result[0] for result in results]
cur.execute("SELECT id, title, description FROM sports_videos WHERE id = ANY(%s)", (ids,))
results = cur.fetchall()
for result in results[0:2]:
    print(result[1], result[2])
