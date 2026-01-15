"""
imports data from remote server every 20 minutes
"""
import os
from pathlib import Path
import dotenv
import schedule
import time
from apsimNGpy.core_utils.database_utils import read_with_pandas
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from loguru import logger

Base = Path(__file__).parent.parent
data_dir = Base / 'data'
data_dir.mkdir(exist_ok=True)
dotenv.load_dotenv()
dbp = os.getenv('SUPABASE_DB_PASSWORD')

passWord = os.getenv('SUPABASE_DB_PASSWORD')
HOST = os.getenv('SUPABASE_DB_HOST')


def download_data():
    url = URL.create(
        drivername="postgresql+psycopg2",
        username="postgres",
        password=passWord,  # raw password is OK here
        host=HOST,
        port=5432,
        database="postgres",
    )

    engine = create_engine(url)
    df = read_with_pandas(table='metrics', db_or_con=engine)
    df.to_csv(data_dir / 'data.csv', index=False)
    logger.info(f'Succeed downloading data, size and shape:{df.shape}')


if __name__ == '__main__':
    schedule.every(30).minutes.do(download_data)

    while True:
        schedule.run_pending()
        sleep_time = (30 * 60) - 20
        time.sleep(sleep_time)
