import os
import json
from collections import OrderedDict
from operator import itemgetter

data_folder = os.path.join(os.path.expanduser("~"), "Data", "research")
words_folder = os.path.join(data_folder, "results/word_list")

class DictDiffer(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values
    """
    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current, self.set_past = set(current_dict.keys()), set(past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)
    def added(self):
        return self.set_current - self.intersect
    def removed(self):
        return self.set_past - self.intersect
    def changed(self):
        return set(o for o in self.intersect if self.past_dict[o] != self.current_dict[o])
    def unchanged(self):
        return set(o for o in self.intersect if self.past_dict[o] == self.current_dict[o])


# [file1, file2, ... ]
def read_files_from_dir():
    """ returns full file paths from tweets data folder """
    files_list = [os.path.join(words_folder, file) for file in os.listdir(words_folder)]
    return files_list


def get_sorted_dict(data):
    d = OrderedDict(sorted(data.items(), key=itemgetter(1), reverse=True))
    return d


def copy_keys(table1, keys):
    """
    table1 -- dict that we copy FROM.
    table2 -- dict that we copy TO.
    """
    table2 = {}
    for key in keys:
        if table1[key]:
            table2[key] = table1[key]
        else:
            print("key does not exist in table 2")
            break
    return table2

files = read_files_from_dir()
python_file = files[9]
control_file = files[4]

with open(python_file, 'r') as inf:
    python_dict = get_sorted_dict(json.load(inf))


with open(control_file, 'r') as inf:
    control_dict = get_sorted_dict(json.load(inf))

# d = DictDiffer(python_words, control_words)

python_list = list(python_dict.keys())
control_list = list(control_dict.keys())

n = [10, 50, 100, 150, 200]

diff = set()
for word_size in n:
    python_top_n_words = python_list[:word_size]
    control_top_n_words = control_list[:word_size]

    diff = set(python_top_n_words) - set(control_top_n_words)
    reverse_diff = set(control_top_n_words) - set(python_top_n_words)

    print("Difference (" + str(word_size) + "): " + str(len(diff)))
    print(diff)
    # print(reverse_diff)

word_list = list(reverse_diff)
word_cloud_dict = dict()
word_cloud_dict = copy_keys(control_dict, word_list)
word_cloud_dict = get_sorted_dict(word_cloud_dict)
print(word_cloud_dict)

import csv

writer = csv.writer(open('control_cloud.csv', 'w'))
for key, value in word_cloud_dict.items():
    writer.writerow([key, value])