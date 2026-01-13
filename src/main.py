import pandas as pd
import dotenv
import os
from apsimNGpy.core_utils.database_utils import read_db_table

dotenv.load_dotenv()
dbp = os.getenv('SUPABASE_DB_PASSWORD')

from sqlalchemy import create_engine
import os
passWord = os.getenv('SUPABASE_DB_PASSWORD')
from sqlalchemy import create_engine
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

url = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password=passWord,   # raw password is OK here
    host="db.elheeekvqchcvvdwlnzc.supabase.co",
    port=5432,
    database="postgres",
)

engine = create_engine(url)
df = read_db_table(db=)