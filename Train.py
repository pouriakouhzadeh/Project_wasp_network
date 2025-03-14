import os
import glob
import shutil
import pickle
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from GeneticAlgoritmOptimization import GeneticAlgorithm
from DataFrameComparer import DataFrameComparer
from logging.handlers import RotatingFileHandler
import gc  # Garbage Collector interface

# Setup enhanced logging with both file and console output
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file_handler = RotatingFileHandler('Train.log', maxBytes=10*1024*1024, backupCount=5)
log_file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_file_handler)
logger.addHandler(console_handler)

# Constants
CURRENCY_FILES = [
    "EURUSD60", "AUDCAD60", "AUDCHF60",
    "AUDNZD60", "AUDUSD60", "EURAUD60",
    "EURCHF60", "EURGBP60", "GBPUSD60",
    "USDCAD60", "USDCHF60"
]

# File directories
INITIAL_CSV_DIRECTORY = "/home/pouria/project/csv_files_initial/"
NEW_CSV_DIRECTORY = "/home/pouria/project/csv_files/"
TEMP_CSV_DIRECTORY = "/home/pouria/project/temp_csv_dir/"

def check_directories():
    paths = [
        "/home/pouria/project/trained_models/",
        "/home/pouria/project/csv_files_initial/",
        "/home/pouria/project/temp_csv_dir/"
    ]
    for path in paths:
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
        if not os.access(path, os.W_OK):
            logger.error(f"No write access to path: {path}")

# Function to process each currency file
def process_currency_file(currency_file, population, NG):
    ga = GeneticAlgorithm()
    result = ga.main(currency_file, population, NG)
    return currency_file, result

# Function to read data from a CSV file
def read_data(file_path, tail_size=21000):
    try:
        data = pd.read_csv(file_path)
        data = data.tail(tail_size)
        data.reset_index(inplace=True, drop=True)
        return data
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        return None

# Function to copy files from one directory to another
def copy_files(source_folder, destination_folder):
    n = 0
    for file_path in glob.glob(os.path.join(source_folder, '*.csv')):
        file_name = os.path.basename(file_path)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copy2(file_path, destination_file)
        n += 1
    logger.info(f"{n} CSV files copied successfully")

# Main execution block
if __name__ == "__main__":
    check_directories()  # بررسی مسیرها و دسترسی‌ها قبل از شروع برنامه اصلی
    number_of_cycle = 1
    while True:
        try:
            logger.info(f"Cycle number = {number_of_cycle}")
            copy_files(NEW_CSV_DIRECTORY, TEMP_CSV_DIRECTORY)
            population, NG = 300, 70
            currency_name_list = []

            for currency in CURRENCY_FILES:
                parameters_file_path = f"{os.path.join('/home/pouria/project/trained_models', currency + '_parameters.pkl')}"
                try:
                    with open(parameters_file_path, 'rb') as file:
                        ind = pickle.load(file)

                    if all(x == 0 for x in ind[:3]):
                        currency_name_list.append(currency)
                    else:
                        df1 = read_data(f"{os.path.join(INITIAL_CSV_DIRECTORY, currency + '.csv')}")
                        df2 = read_data(f"{os.path.join(TEMP_CSV_DIRECTORY, currency + '.csv')}", tail_size=ind[3])
                        if df1 is not None and df2 is not None:
                            comparer = DataFrameComparer(df1, df2)
                            percent_change = comparer.My_compare_dataframes()
                            logger.info(f"Percent change for {currency} is {percent_change}")
                            if percent_change >= 3:
                                currency_name_list.append(currency)
                        del df1, df2
                        gc.collect()
                except Exception as e:
                    currency_name_list.append(currency)
                    logger.error(f"Cannot read parameters file for {currency}, added to list for training. Error: {e}")

            logger.info(f"Currency list length: {len(currency_name_list)}")
            concurrency = len(currency_name_list)
            logger.info(f"Concurrency level: {concurrency}")

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(process_currency_file, currency, population, NG): currency for currency in currency_name_list}
                for future in as_completed(futures):
                    currency_file, result = future.result()
                    logger.info(f"Result for {currency_file} = {result}")

            logger.info(f"The cycle {number_of_cycle} finished. Starting next one...")
            number_of_cycle += 1

        except Exception as e:
            logger.critical(f"An unexpected error occurred during cycle {number_of_cycle}: {e}")
            break  # Optionally, stop the loop if a critical error occurs
