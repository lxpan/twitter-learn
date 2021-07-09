# twitter-authorship-prediction

## What this is for
This project investigates the problem of authorship attribution -- namely, given different bodies of text, can we predict "which author wrote what text"? Historically, this question has often been asked regarding the works of Shakespeare -- were his plays written by one person, or by a number of people? In this case, we look at tweets obtained from Twitter, which are known for their short length (140 characters or less).

## What makes this unique
Instead of randomly sampling a list of tweets and authors, we sample tweets based on a keyword. This keyword is the word "Python", and we perform disambiguation to ensure that they are talking about "Python" the programming language, and not "python" the animal. We then compare the performance of performing authorship attribution using our 'Python enthusiasts' vs a control group of randomly sampled users.

## How to best view the code
Jupyter notebooks can take a while to load in Github, so it's recommended to either view them in Jupyter Lab, or use Visual Studio Code with the Python and Jupyter notebook extensions installed.

## Overview of the code
The following is a summary of what each Jupyter notebook does:
- `1-get_twitter.ipynb`: Download a list of tweet IDs using the Twitter API
- `2-label_twitter.ipynb`: Manually label tweets — are they referring to "Python" the language or the snake? This forms our dataset to perform disambiguation
- `3-classify_twitter.ipynb`: Train a naive bayes model that can tell the difference between "Python" the language vs "python" the animal
- `4-get_users.ipynb`: Get list of users that have tweets about "Python". Our model from the previous step will ensure only tweets about "Python" the programming language get through
- `5-get_users_control.ipynb`: Get a list of users that form the control group (their tweets can be about anything)
- `6-create_replicable_dataset.ipynb`: Combine the tweets and their class labels into a single JSON file
- `7-replicate_dataset.ipynb`: Download the tweets using the Twitter API, based on the tweet IDs collected in the previous step
- `8-research_train_test.ipynb`: Using the processed twitter dataset, train a model to predict the authors of the tweets
- `9-plot_graphs.ipynb`: Graph the performance of the model
- `10-word_comparison.py`: Compare overlap of vocabulary. Do our 'Python enthusiasts' use more words than our control group?

## References
The paper associated with this project can be found [here](https://books.google.com.au/books?id=k64sDwAAQBAJ&pg=PA250&lpg=PA250&dq=Improving+Authorship+Attribution+in+Twitter+Through+Topic-Based+Sampling&source=bl&ots=DmnFH6sruC&sig=ACfU3U37hcocinVmMNu_THLCKbC9n1b_gA&hl=en&sa=X&ved=2ahUKEwicvpOzi9XxAhWr-nMBHWBYA-IQ6AEwBXoECAkQAw#v=onepage&q=Improving%20Authorship%20Attribution%20in%20Twitter%20Through%20Topic-Based%20Sampling&f=false)


