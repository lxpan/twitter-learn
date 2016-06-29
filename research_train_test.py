"""
Issues: 
* rand_users() sampling duplicate users (should be solved)
* returning users less than 120 tweets (solved)
* not enough users to make 100 (solved)

To do:
* batch reading of json files (done)
* for users with more than 120 tweets, only consider first 120
"""
import os
import json
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from collections import OrderedDict
from operator import itemgetter

data_folder = os.path.join(os.path.expanduser("~"), "Data", "research")
# results_output_filename = os.path.join(data_folder, "attribution_results_control_2.json")
tweets_folder = os.path.join(data_folder, "control_tweets")
#tweets_folder = os.path.join(data_folder, "tweets")

results_folder = os.path.join(data_folder, "results")


# [file1, file2, ... ]
def read_files_from_dir(datafolder):
    """ returns full file paths from tweets data folder """
    files_list = [os.path.join(tweets_folder, file) for file in os.listdir(tweets_folder)]
    return files_list

# read all tweet files from a folder
files_list = read_files_from_dir(tweets_folder)


def read_merge_tweets(files):
    """ Given an array of files, read each file (and concatenate?)"""
    dicts = []
    for file in files:
        with open(file, 'r') as inf:
            tweets_dict = json.load(inf)
            tweets_processed = remove_low_tweet_authors(tweets_dict)
            # tweets = remove_non_ascii_chars(tweets)
            dicts.append(tweets_processed)
    # merge step
    super_dict = defaultdict(list)  # uses set to avoid duplicates
    for d in dicts:
        for k, v in iter(d.items()):
            super_dict[k].extend(v)
    return super_dict


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


def rand_users(users, sample_size):
    sampled = random.sample(users, sample_size)
    return sampled


def remove_hashtag(tweet):
    pass


def remove_at_symbol(tweet):
    pass


def remove_low_tweet_authors(tweets):
    new_dict = {}
    """ 120 is an optimal number of tweets for authorship attribution (Layton)"""
    for key in tweets:
        if len(tweets[key]) < 120:
            # tweets.pop(key, None)
            continue
        else:
            new_dict[key] = tweets[key]
    return new_dict

# tweets.keys()


# In[10]:

# print(tweets_folder)
for file in files_list:
    print(file)
tweets = read_merge_tweets(files_list)
print(len(tweets.keys()))


# In[11]:

# authors = {}  ## not needed for actual data mining
def join_documents(tweets):
    """ In Python 3, iteritems() has been replaced simply with items() """
    documents = []
    classes = []
    author_num = 0
    # use sorted() to enforce ordered dict iteration
    for key, value in iter(sorted(tweets.items())):
        # concatenate documents into one giant corpus
        documents.extend(value)
        # assign classes values to each respective authors' tweets
        classes.extend([author_num] * len(value))
        author_num += 1
        # print("Author: " + key + ", tweets: " + str(len(value)))
    return documents, classes

restricted_words = ['http', 'rt']


def get_unique_words(document, lower=False):
    words = defaultdict(int)
    tweet_length = []
    for tweet in document:
        line = tweet.split()
        num_words = len(line)
        tweet_length.append(num_words)
        # append any unique words (to whole corpus) found in tweet
        for word in line:  # word in tweet will return chars, therefore use line
            if lower is True:
                word = word.lower()
            if word.isalpha():
                if not words[word] and word not in restricted_words:
                    words[word] = 1
                elif words[word] >= 1:
                    words[word] += 1
            else:
                continue
    average_tweet_length = np.mean(tweet_length)
    return words, average_tweet_length


def get_sorted_dict(data):
    d = OrderedDict(sorted(data.items(), key=itemgetter(1), reverse=True))
    return d


# for each key, sum values across both dicts
def sum_dicts(first, second):
    """ Warning: very slow! """
    from collections import Counter
    first = Counter(first)
    second = Counter(second)
    first_plus_second = first + second
    return dict(first_plus_second)

from sklearn.svm import SVC # support vector machines
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import grid_search

""" Set up the parameters. 'C' "refers to how much classifier should aim to predict all training samples correctly"
Kernel introduces non-linear elements to make them linearly separable(?) 
"""
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svr = SVC()
grid = grid_search.GridSearchCV(svr, parameters)

# extract character ngrams
pipeline = Pipeline([('feature_extraction', 
                      CountVectorizer(analyzer='char', ngram_range=(3, 3))),
                     ('classifier', grid)])

scores = defaultdict(list)

iter_sample = [20, 30]
RUNS = 30

for sample_size in iter_sample:
    vocab_size = []
    tweet_size = []
    master_word_dict = {}
    master_vocab_dict = {}
    count = 0
    while count < RUNS:
        author_subset = rand_users(tweets.keys(), sample_size)
        tweets_subset = copy_keys(tweets, author_subset)
        documents, classes = join_documents(tweets_subset)
        """ calculate average tweet lengths and author vocab sizes
        words, avg_tweet_length = get_unique_words(documents, lower=True)
        tweet_size.append(avg_tweet_length)
        vocab_size.append(len(words))
        master_word_dict = sum_dicts(master_word_dict, words)
        """

        # print("Vocab size: " + str(vocab_size[count]))
        # retrieve features (ngrams)
        # NB: ngrams are not as likely to be useful compared to words


        print("Creating model ...")
        model = pipeline.fit(documents, classes)
        print("Model created")
        feature_set = model.named_steps['feature_extraction']
        print("Features extracted")
        master_vocab_dict = sum_dicts(master_vocab_dict, feature_set.vocabulary_)
        print("Dicts summed")
        print("Pass: " + str(count))


        """ calculate scores via 3-fold (default) cross validation
        score = cross_val_score(pipeline, documents, classes, scoring='f1')
        avg_score = np.mean(score)
        scores[sample_size].append(avg_score)
        print("Run: " + str(count + 1) + ", Samples: " + str(sample_size) + ", Score: " + str(avg_score))
        """
        count += 1
    # sort master word dict
    # words_sorted = get_sorted_dict(master_word_dict)
    # word_list = list(words_sorted.keys())
    # master_word_dict = get_sorted_dict(master_word_dict)
    master_vocab_dict = get_sorted_dict(master_vocab_dict)
    # turn OrderedDict into list
    vocab_list = [(k, str(v)) for k, v in master_vocab_dict.items()]

    """ End of run summary """
    print("Samples: " + str(sample_size) + ", sruns: " + str(RUNS))
#    print("Score: {:.3f}".format(np.mean(scores[sample_size])))
#    print("Mean vocab size: " + str(np.mean(vocab_size)))
#    print("Average tweet size: " + str(np.mean(tweet_size)))

    file_name = "control_ngrams_" + str(sample_size) + ".json"
    word_list_output_filename = os.path.join(results_folder, file_name)
    # save what we currently have

    with open(word_list_output_filename, 'w') as fp:
        json.dump(vocab_list, fp)
        print(file_name + " written")


# In[13]:

# get_ipython().magic('matplotlib inline')