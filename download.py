# import pandas as pd
import shutil, os
import os.path as osp
import boto3
import botocore
from botocore.handlers import disable_signing
import tqdm
from botocore import UNSIGNED
from botocore.client import Config
import hashlib
import zenodo_get as zget
import zipfile
import shutil
from pathlib import Path
from enum import Enum
import sys

base_URL = "milacomplexitydataset"
zen_id = 7008205
# https://zenodo.org/record/7008205#.YxtIwi0r1hC
base_directory = "TG_network_datasets"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DataSetName(Enum):
    CanParl = "CanParl"
    Contacts = "Contacts"
    Enron = "enron"
    Flights = "Flights"
    Lastfm = "lastfm"
    Mooc = "mooc"
    Reddit = "reddit"
    SocialEvo = "SocialEvo"
    UCI = "uci"
    UNtrade = "UNtrade"
    UNvote = "UNvote"
    USLegis = "USLegis"
    Wikipedia = "wikipedia"


sub_dict = {
    "CanParl": ["CanParl.csv", "ml_CanParl.csv", "ml_CanParl.npy", "ml_CanParl_node.npy"],
    "Contacts": ["Contacts.csv", "ml_Contacts.csv", "ml_Contacts.npy", "ml_Contacts_node.npy"],
    "enron": ["ml_enron.csv", "ml_enron.npy", "ml_enron_node.npy"],
    "Flights": ["Flights.csv", "ml_Flights.csv", "ml_Flights.npy", "ml_Flights_node.npy"],
    "lastfm": ["lastfm.csv", "ml_lastfm.csv", "ml_lastfm.npy", "ml_lastfm_node.npy"],
    "mooc": ["ml_mooc.csv", "ml_mooc.npy", "ml_mooc_node.npy", "mooc.csv"],
    "reddit": ["ml_reddit.csv", "ml_reddit.npy", "ml_reddit_node.npy", "reddit.csv"],
    "SocialEvo": ["ml_SocialEvo.csv", "ml_SocialEvo.npy", "ml_SocialEvo_node.npy"],
    "uci": ["ml_uci.csv", "ml_uci.npy", "ml_uci_node.npy"],
    "UNtrade": ["ml_UNtrade.csv", "ml_UNtrade.npy", "ml_UNtrade_node.npy", "UNtrade.csv"],
    "UNvote": ["ml_UNvote.csv", "ml_UNvote.npy", "ml_UNvote_node.npy", "UNvote.csv"],
    "USLegis": ["ml_USLegis.csv", "ml_USLegis.npy", "ml_USLegis_node.npy", "USLegis.csv"],
    "wikipedia": ["ml_wikipedia.csv", "ml_wikipedia.npy", "ml_wikipedia_node.npy", "wikipedia.csv"]
}

data_set_list = list[DataSetName]


def unzip_delete():
    with zipfile.ZipFile("TG_network_datasets.zip", 'r') as zip_ref:
        zip_ref.extractall()
    try:
        os.remove("TG_network_datasets.zip")
    except OSError:
        pass
    dirpath = Path('__MACOSX')
    if dirpath.exists():
        shutil.rmtree(dirpath)
    try:
        os.remove("md5sums.txt")
    except OSError:
        pass


class TemporalDataSets(object):
    def __init__(self, data_list: data_set_list = None):
        """
            - data_list (data_set_list): [ list of dataset enums ]
        """
        if not data_list:
            self.data_list = [
                DataSetName.CanParl,
                DataSetName.Contacts,
                DataSetName.Enron,
                DataSetName.Flights,
                DataSetName.Lastfm,
                DataSetName.Mooc,
                DataSetName.Reddit,
                DataSetName.SocialEvo,
                DataSetName.UCI,
                DataSetName.UNtrade,
                DataSetName.UNvote,
                DataSetName.USLegis,
                DataSetName.Wikipedia
            ]
        else:
            self.data_list = data_list  # original name
            sys.stdout.write("Current Data selectively loaded: ")
            for dataset in self.data_list:
                sys.stdout.write(f"{str(dataset)} ")
            sys.stdout.write("\n")

        self.check_downloaded()

    def download_file(self):
        print("Data missing, download recommended")
        inp = input('Will you download the dataset(s) now? (y/N)\n').lower()
        if inp == 'y':
            print(f"{bcolors.WARNING}Download started, this might take a while . . . {bcolors.ENDC}")
            os.system('zenodo_get ' + str(zen_id))
            unzip_delete()
        else:
            print("Download cancelled")

    def check_downloaded(self):
        if not osp.isdir(f"./{base_directory}"):
            print(f"dict: {base_directory} not found")
            self.download_file()
        list_data_not_found = []
        for data_set_name in self.data_list:
            data_found = True
            for file_name in sub_dict[str(data_set_name.value)]:
                path = f"./{base_directory}/{str(data_set_name.value)}/{file_name}"
                if not Path(path).exists():
                    data_found = False
            if not data_found:
                list_data_not_found.append(data_set_name.value)
        if not list_data_not_found:
            print("All data found")
        else:
            sys.stdout.write("The following datasets not found ")
            for data_set_name in list_data_not_found:
                sys.stdout.write(f"{data_set_name} ")
            sys.stdout.write("\n")
            self.download_file()

    def redownload(self):
        inp = input('Confirm re-download? (y/N)\n').lower()
        if 'y' == inp:
            os.system('zenodo_get ' + str(zen_id))
            unzip_delete()
        else:
            print("download cancelled")

    def process(self):
        pass
        # need to implement the processing.


# DREAM
# processing method, for the dataset.
# data loader -> show how to use the data loader with the 5 methods.

# Goals
# make it work with the https://zenodo.org/record/7008205#.YxtIwi0r1hC instead of AWS.
# Implement data-processing, -> training data.


if __name__ == "__main__":
    input_list = [DataSetName.CanParl]
    example_data = TemporalDataSets(data_list=input_list)
    example_data.redownload()
    example_data.process()


# notes for today, upload it to github.
# share link to with the chat.
# TODO
# make sure it works for indervidual files download
# finish the processing function
# finish the evaluation function - edgecase-testing errors for the function.
# finish a pip install setup