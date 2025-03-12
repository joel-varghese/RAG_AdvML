"""
Simple RAG

1. set up postgres connection
2. get question from user
3. Use question to search postgres table
4. Format the results in a LLM-friendly way
5. Send results to the LLM

"""

import os

import numpy as np
import psycopg2
from dotenv import load_dotenv
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

# get question from user
question = "ball"

cur.execute("SELECT * FROM sports_videos WHERE title LIKE %s OR description LIKE %s", (f"%{question}%", f"%{question}%"))

results = cur.fetchall()
print(results)