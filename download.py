import os
import os.path as osp
import random
import shutil
import sys
import zipfile
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from clint.textui import progress


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


zen_id = 7213796
zend_id_all = 7008205
# https://zenodo.org/record/7008205#.YxtIwi0r1hC
base_directory = "TG_network_datasets"


# zen_id = 7008204 for the most updated version.


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


check_dict = {
    "canparl": DataSetName.CanParl,
    "contacts": DataSetName.Contacts,
    "enron": DataSetName.Contacts,
    "flights": DataSetName.Flights,
    "lastfm": DataSetName.Lastfm,
    "mooc": DataSetName.Mooc,
    "reddit": DataSetName.Reddit,
    "socialEvo": DataSetName.SocialEvo,
    "UCI": DataSetName.UCI,
    "un_trade": DataSetName.UNtrade,
    "un_vote": DataSetName.UNvote,
    "us_Legis": DataSetName.USLegis,
    "wikipedia": DataSetName.Wikipedia,
}

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
    os.remove("./md5sums.txt") if os.path.exists("./md5sums.txt") else None

    for filename in Path(".").glob("*.tmp"):
        filename.unlink()

    if not os.path.exists("./TG_network_datasets.zip"):
        print(f"{bcolors.FAIL}DOWNLOAD FAILED{bcolors.ENDC}, TG_network_datasets not found")
        return

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
    def __init__(self, data_list: str = None, data_set_statistics: bool = True):
        """
            - data_list (data_set_list): [ list of dataset enums ]
        """

        self.data_set_statistics = data_set_statistics
        self.url = f"https://zenodo.org/record/{zen_id}"
        self.mask = None
        if data_list not in check_dict.keys():

            sys.stdout.write(f" input for TemporalDataSets: '{str(data_list)}' not found, \n inputs must be "
                             f"from the following list: \n")
            for key in check_dict.keys():
                sys.stdout.write(f"   {key}\n")

            inp = input('Exit program ? this is recommended action (y/N)').lower()
            if inp == "y":
                exit()
            else:
                sys.stdout.write(bcolors.WARNING + "program will continue but program is unsafe \n" + bcolors.ENDC)

        else:
            self.data_list = [check_dict.get(data_list)]  # original name
            # sys.stdout.write("Dataset title: ")
            # for dataset in self.data_list:
            #    sys.stdout.write(f"{str(dataset)}")
            # sys.stdout.write("\n")
            self.url += f"/files/{self.data_list[0].value}.zip?download=1"
            self.path_download = f"./{self.data_list[0].value}.zip"

        self.check_downloaded()

    def delete_single(self):
        try:
            os.remove(f"{self.data_list[0].value}.zip")
        except OSError:
            pass
        dirpath = Path('__MACOSX')
        if dirpath.exists():
            shutil.rmtree(dirpath)
        try:
            os.remove("md5sums.txt")
        except OSError:
            pass

    def download_file(self):
        print("Data missing, download recommended!")
        inp = input('Will you download the dataset(s) now? (y/N)\n').lower()
        if inp == 'y':
            print(f"{bcolors.WARNING}Download started, this might take a while . . . {bcolors.ENDC}")
            print(f"Dataset title: {self.data_list[0].value}")
            r = requests.get(self.url, stream=True)
            with open(self.path_download, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                    if chunk:
                        f.write(chunk)
                        f.flush()

            os.makedirs(f"./{base_directory}", exist_ok=True)

            try:
                shutil.rmtree(f"./{base_directory}/{self.data_list[0].value}")
            except:
                pass

            with zipfile.ZipFile(self.path_download, 'r') as zip_ref:
                zip_ref.extractall(f"./{base_directory}")

            self.delete_single()
            print(f"{bcolors.OKGREEN}Download completed {bcolors.ENDC}")

        else:
            print(bcolors.FAIL + "Download cancelled" + bcolors.ENDC)

    def check_downloaded(self):
        if not osp.isdir(f"./{base_directory}"):
            print(f"dict: {base_directory} not found")
            self.download_file()
            return
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
            sys.stdout.write("The following datasets not found: ")
            for data_set_name in list_data_not_found:
                sys.stdout.write(f"{data_set_name} ")
            sys.stdout.write("\n")
            self.download_file()

    def redownload(self):
        print(
            bcolors.WARNING + "attempting redownload, will remove ALL files and download ALL possible files" + bcolors.ENDC)
        inp = input('Confirm redownload? (y/N)\n').lower()
        if 'y' == inp:
            try:
                shutil.rmtree(f"./{base_directory}")
            except:
                pass
            try:
                _ = os.system('zenodo_get ' + str(zend_id_all))
            except KeyboardInterrupt:
                pass
            except Exception as e:
                raise

            unzip_delete()
        else:
            print("download cancelled")

    def process(self):
        # f"./{base_directory}/self."
        # need to implement the processing.
        self.get_split()
        self.node_features, self.edge_features, self.full_data, self.train_data, self.val_data, self.test_data, \
        self.new_node_val_data, self.new_node_test_data, masks = self.get_data()

        self.mask = masks

        # todo save these self.node_features.

    '''
    this function will 
    1. retrieve the saved splits for node masks, train, val, test if default=True
    2. generate new masks if default = False, and use the random seed to generate the new masks
    '''

    def get_split(self, default=True, seed=2020):
        if (default):
            train_idx = np.load(...)
            val_idx = np.load(...)
            test_idx = np.load(...)
            self.masks = {"idx_train": train_idx, "idx_val": val_idx, "idx_test": test_idx}
        else:
            # generate the random split here
            new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    def get_data(self, different_new_nodes_between_val_and_test=False, randomize_features=False):

        if (self.mask is None):
            sys.stdout.write(f"{bcolors.FAIL}self.masks not found error{bcolors.ENDC}")
            raise Exception("Please run get_split() before get_data()")

        train_idx = self.masks["idx_train"]
        val_idx = self.masks["idx_val"]
        test_idx = self.masks["idx_test"]
        new_node_val_idx = self.masks['idx_new_node_val']
        new_node_test_idx = self.masks['idx_new_node_test']

    def get_data(self, val_ratio=0.15, test_ratio=0.15, different_new_nodes_between_val_and_test=False,
                 randomize_features=False):

        value = self.data_list[0].value
        path_to_dict = f"./{base_directory}/{value}"

        PATH = f"{path_to_dict}/{value}.csv"
        OUT_DF = f"{path_to_dict}/ml_{value}.csv"
        OUT_FEAT = f"{path_to_dict}/ml_{value}.npy"
        OUT_NODE_FEAT = f"{path_to_dict}/ml_{value}_node.np"

        graph_df = pd.read_csv(OUT_DF)
        edge_features = np.load(OUT_FEAT)
        node_features = np.load(OUT_NODE_FEAT)

        # additional for CAW data specifically
        if dataset_name in ['enron', 'socialevolve', 'uci']:
            node_zero_padding = np.zeros((node_features.shape[0], 172 - node_features.shape[1]))
            node_features = np.concatenate([node_features, node_zero_padding], axis=1)
            edge_zero_padding = np.zeros((edge_features.shape[0], 172 - edge_features.shape[1]))
            edge_features = np.concatenate([edge_features, edge_zero_padding], axis=1)

        if randomize_features:
            node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

        node_set = set(sources) | set(destinations)
        n_total_unique_nodes = len(node_set)

        # Compute nodes which appear at test time
        test_node_set = set(sources[timestamps > val_time]).union(
            set(destinations[timestamps > val_time]))
        # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
        # their edges from training

        # where 'randomness' matters...
        new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

        # Mask saying for each source and destination whether they are new test nodes
        new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

        # Mask which is true for edges with both destination and source not being new test nodes (because
        # we want to remove all edges involving any new test node)
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

        # For train we keep edges happening before the validation time which do not involve any new node
        # used for inductiveness
        train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

        train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                          edge_idxs[train_mask], labels[train_mask])

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_data.sources).union(train_data.destinations)
        assert len(train_node_set & new_test_node_set) == 0
        new_node_set = node_set - train_node_set

        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        if different_new_nodes_between_val_and_test:
            n_new_nodes = len(new_test_node_set) // 2
            val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
            test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

            edge_contains_new_val_node_mask = np.array(
                [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
            edge_contains_new_test_node_mask = np.array(
                [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


        else:
            edge_contains_new_node_mask = np.array(
                [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

        # validation and test with all edges
        val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                        edge_idxs[val_mask], labels[val_mask])

        test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                         edge_idxs[test_mask], labels[test_mask])

        # validation and test with edges that at least has one new node (not in training set)
        new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                                 timestamps[new_node_val_mask],
                                 edge_idxs[new_node_val_mask], labels[new_node_val_mask])

        new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                                  timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                                  labels[new_node_test_mask])

        if self.data_set_statistics:
            print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                         full_data.n_unique_nodes))
            print("The training dataset has {} interactions, involving {} different nodes".format(
                train_data.n_interactions, train_data.n_unique_nodes))
            print("The validation dataset has {} interactions, involving {} different nodes".format(
                val_data.n_interactions, val_data.n_unique_nodes))
            print("The test dataset has {} interactions, involving {} different nodes".format(
                test_data.n_interactions, test_data.n_unique_nodes))
            print("The new node validation dataset has {} interactions, involving {} different nodes".format(
                new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
            print("The new node test dataset has {} interactions, involving {} different nodes".format(
                new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
            print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
                len(new_test_node_set)))

        return node_features, edge_features, full_data, train_data, val_data, test_data, \
               new_node_val_data, new_node_test_data, {"idx_train": train_mask, "idx_test": test_mask,
                                                       "idx_val": val_mask}


# DREAM
# processing method, for the dataset.
# data loader -> show how to use the data loader with the 5 methods.

# Goals
# make it work with the https://zenodo.org/record/7008205#.YxtIwi0r1hC instead of AWS.
# Implement data-processing, -> training data.


if __name__ == "__main__":
    random.seed(2020)
    input_list = "canparl"
    example_data = TemporalDataSets(data_list=input_list)






    #example_data.redownload()
    # example_data.process()

    #training_data = example_data.train_data
    #training_data = example_data.training_data
    #val_data = example_data.val_data

# TODO
# make sure it works for indervidual files download
# finish the processing function
# finish the evaluation function - edgecase-testing errors for the function.
# finish a pip install setup