"""
This script is used to extract data from papers in parallel using MPI.
"""
import datetime
import logging
import os
import signal
import sys
import time
import traceback

from mpi4py import MPI
from tadf_models.models import PhotoluminescenceWavelength
from chemdataextractor.model import ThemeCompound

from tadf_model_extractor import TADFExtractor


class TimeoutException(Exception):
    """
    Exception raised when a function takes too long to execute, used in signal.alarm.
    """
    pass


def timeout_handler(signum, frame):
    """
    Handler for the signal.alarm, raises TimeoutException.
    """
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)

AWAITING_DATA_TAG = 111
RETURNING_DATA_TAG = 222
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
is_main_thread = rank == 0

logger = logging.getLogger("mpi_extract")
logger.setLevel(logging.INFO)

if is_main_thread:
    file_handler = logging.FileHandler("mpi_extract.log", mode="w")
else:
    file_handler = logging.FileHandler("mpi_extract.log", mode="a")
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define log format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def customized_files():
    """
    Returns a customized list of files to be extracted.
    """
    files = []
    return files


def extract_paper(
    paper_root: str,
    save_root: str,
    dbname: str,
    file_n: str,
    models=(
        ThemeCompound,
        PhotoluminescenceWavelength,
    ),
):
    """
    Extracts data from a paper and saves it to the specified location.

    Parameters
    ----------
    paper_root : str
        The root directory where the paper files are located.
    save_root : str
        The directory where the extracted data will be saved.
    dbname : str
        The name of the database file.
    file_n : str
        The name of the file to be extracted, just the name not path.
    models : tuple, optional
        A tuple of models to be used for extraction, by default [PhotoluminescenceWavelength]
    """
    doc_start_time = datetime.datetime.now()
    signal.alarm(1800)
    try:
        mdb = TADFExtractor(
            paper_root=paper_root,
            save_root=save_root,
            save_filename=dbname,
            target_paper=file_n,
            models=models
        )
        mdb.extraction()

        with open(f"{save_root}/completed.txt", "a", encoding="utf8") as cmpl:
            cmpl.write(file_n + "\n")
    except TimeoutException:
        logger.warning("File %s took too long to extract, skipping the file.", file_n)

        with open(f"{save_root}/Timeout_error_files.txt", "a+", encoding="utf8") as error_file:
            error_file.write(file_n + "\n")
        with open(f"{save_root}/Other_error_files.txt", "a+", encoding="utf8") as cmpl:
            cmpl.write(file_n + "\n")
    except RecursionError:
        with open(f"{save_root}/Recursion_error_files.txt", "a+", encoding="utf8") as error_file:
            error_file.write(file_n + "\n")
    except AttributeError as e2:
        logger.error("%s Attribute Error %s occurred.", file_n, e2)

        with open(f"{save_root}/completed.txt", "a+", encoding="utf8") as cmpl:
            cmpl.write(file_n + "\n")
    except RuntimeError:
        with open(f"{save_root}/CUDA_error_files.txt", "a+", encoding="utf8") as error_file:
            error_file.write(file_n + "\n")
        logger.error(traceback.format_exc())

    except Exception as unexpected:
        with open(f"{save_root}/Other_error_files.txt", "a+", encoding="utf8") as cmpl:
            cmpl.write(file_n + "\n")
        logger.error("######## Unexpected Error ########")
        logger.error("%s", file_n)
        logger.error(traceback.format_exc())
        logger.error(unexpected)
        logger.error("##################################")

    finally:
        signal.alarm(0)
    doc_end_time = datetime.datetime.now()

    logger.info("%s took: %s", file_n, doc_end_time - doc_start_time)


def extract_mpi(paper_root, save_root, dbname, start=0, end=None):

    if is_main_thread:
        logger.info("%d processes running.", size)
        logger.debug("starting from %d-th, ending at %d-th.", start, end)

        all_start_time = datetime.datetime.now()
        index = 0
        n_finished = 0
        try:
            with open(f"{save_root}/completed.txt", "r", encoding="utf8") as f:
                logger.info("Reading extracted files for skipping ...")

                completed = f.read().split("\n")
        except FileNotFoundError:
            completed = []
        all_filenames = [
            filename
            for filename in os.listdir(paper_root)
            if filename.endswith("ml") or filename.endswith("md")
        ][start:end]
        logger.debug("num papers in paper_root: %s", len(os.listdir(paper_root)))
        # all_filenames = customized_files()[start:end]
        new_filenames = [
            filename for filename in all_filenames if filename not in completed
        ]
        num_papers = len(new_filenames)
        logger.info("%d new papers in total.", num_papers)
        
        if num_papers == 0:
            logger.info("No new papers to extract.")
            logger.info(
                "Extraction as a whole took: %s", datetime.datetime.now() - all_start_time
            )
            comm.Abort()
            
        time.sleep(0.5)

        while True:
            status = MPI.Status()

            if index == 0:
                for i in range(1, min(num_papers + 1, size)):
                    # send the first batch of papers to i-th workers
                    filename = new_filenames[index]
                    logger.debug("Extracting %s on worker %d", filename, i)
                    send_to_worker(i, filename, paper_root, save_root, dbname)
                    index += 1
            else:
                result = comm.recv(
                    source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
                )
                finished_worker_index = status.Get_source()
                n_finished += 1
                logger.debug("Worker %d finished, %d papers extracted", finished_worker_index, n_finished)
                
                if n_finished == index and index >= num_papers:
                    logger.info("All papers distributed and extracted.")
                    break
                elif index < num_papers:
                    logger.debug("Extracting %s on worker %d", filename, finished_worker_index)
                    filename = new_filenames[index]
                    send_to_worker(
                        finished_worker_index, filename, paper_root, save_root, dbname
                    )
                    index += 1
        for i in range(1, size):
            exit_worker(i)
        logger.info(
            "Extraction as a whole took: %s", datetime.datetime.now() - all_start_time
        )

    else:
        time.sleep(0.5)
        start_worker()


def send_to_worker(worker_index, target_paper, paper_root, save_root, dbname):
    data = {
        "exit": False,
        "target_paper": target_paper,
        "paper_root": paper_root,
        "save_root": save_root,
        "dbname": dbname,
    }
    comm.send(data, dest=worker_index, tag=AWAITING_DATA_TAG)


def exit_worker(worker_index):
    data = {"exit": True}
    comm.send(data, dest=worker_index, tag=AWAITING_DATA_TAG)


def start_worker():
    while True:
        data = comm.recv(source=0, tag=AWAITING_DATA_TAG)
        if data["exit"]:
            break
        target_paper = data["target_paper"]
        paper_root = data["paper_root"]
        save_root = data["save_root"]
        dbname = data["dbname"]
        extract_paper(
            paper_root=paper_root,
            save_root=save_root,
            dbname=dbname,
            file_n=target_paper,
        )
        comm.send([], dest=0, tag=RETURNING_DATA_TAG)
        # break


if __name__ == "__main__":
    # BUILD_NUM = 94
    tadf_paper_root = sys.argv[1]
    tadf_save_root = sys.argv[2]
    start = sys.argv[3]
    end = sys.argv[4]
    output_dir = sys.argv[5]
    if is_main_thread and not os.path.isdir(tadf_save_root):
        logger.info("Save root not found, creating one.")
        os.mkdir(tadf_save_root)
    tadf_save_root = os.path.join(tadf_save_root, output_dir)
    if is_main_thread and not os.path.isdir(tadf_save_root):
        logger.info("Output directory not found, creating one.")
        os.mkdir(tadf_save_root)
    extract_mpi(tadf_paper_root, tadf_save_root, output_dir, int(start), int(end))
