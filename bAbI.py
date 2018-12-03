import os
import re
from functools import reduce
import numpy as np
import random


# Data handler
class Read:
    def __init__(self, directory="tasks_1-20_v1-2/en-valid-10k", train_task="all", valid_task="all",
                 test_task="separate", max_supporting=None):
        # Data
        self.vocab = []
        self.dictionary = {}
        self.train_stories = []
        self.test_stories = []
        self.valid_stories = []

        # Data metrics
        self.vocab_size = 0
        self.supports_max_num = 0
        self.support_max_len = 0
        self.question_max_len = 0

        # Data length
        self.train_stories_length = 0
        self.valid_stories_length = 0
        self.test_stories_length = 0

        # Load
        self.load(directory=directory, train_task=train_task, valid_task=valid_task, test_task=test_task,
                  max_supporting=max_supporting)

        # Variable for iterating batches
        self.shuffled_training_indices = list(range(self.train_stories_length))
        random.shuffle(self.shuffled_training_indices)
        self.batch_begin = 0
        self.epoch_complete = False

    def iterate_batch(self, batch_dim):
        # Reset and shuffle batch when all items have been iterated
        if self.epoch_complete:
            # Reset batch index
            self.batch_begin = 0

            # Shuffle data
            random.shuffle(self.shuffled_training_indices)

            # Epoch
            self.epoch_complete = True

        # Index of the end boundary of this batch
        batch_end = min(self.batch_begin + batch_dim, self.train_stories_length)

        # Batch
        batch = {item: self.train_stories[item][self.shuffled_training_indices[self.batch_begin:batch_end]]
                 for item in self.train_stories}

        # Update batch index
        self.batch_begin = batch_end

        # Update epoch
        self.epoch_complete = self.batch_begin > self.train_stories_length - batch_dim

        # Return
        return batch

    def read_valid(self, batch_size=None):
        if batch_size:
            indices = random.sample(range(self.valid_stories_length), batch_size)
            return {item: self.valid_stories[item][indices] for item in self.valid_stories}
        else:
            return self.valid_stories

    def read_test(self, task, batch_size=None):
        task = task - 1
        if batch_size:
            indices = random.sample(range(self.test_stories_length), batch_size)
            return {item: self.test_stories[task][item][indices] for item in self.test_stories[task]}
        else:
            return self.test_stories[task]

    def tokenize(self, sent):
        """Return the tokens of a sentence including punctuation.
        tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        """
        return [x.strip() for x in re.findall(r"[\w']+|[.,!?;]", sent) if x.strip()]  # TODO: don't need punctuation

    def parse_stories(self, lines, only_supporting=False, max_supporting=None):
        """Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        """
        data = []
        story = []
        for line in lines:
            line = str.lower(line)
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]

                if max_supporting is not None:
                    substory = substory[-max_supporting:]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
        return data

    def get_stories(self, f, unify_supporting=False, only_supporting=False, max_supporting=None):
        """Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        """
        with open(f) as f:
            data = self.parse_stories(f.readlines(), only_supporting=only_supporting, max_supporting=max_supporting)

        if unify_supporting:
            flatten = lambda data: reduce(lambda x, y: x + y, data)
            data = [(flatten(story), q, answer) for story, q, answer in data]

        return data

    def load(self, directory, train_task="all", valid_task="all", test_task="separate", unify_supporting=False,
             only_supporting=False, max_supporting=None):
        files = [os.path.join(directory, file) for file in os.listdir(directory)]

        print("Extracting stories for training on task:", train_task)
        print("Extracting stories for validation on task:", valid_task)
        print("Extracting stories for testing on task:", test_task)

        for file in files:
            if "train" in file:
                if train_task == "all":
                    self.train_stories += self.get_stories(file, unify_supporting, only_supporting, max_supporting)
                elif "qa{}_".format(train_task) in file:
                    self.train_stories = self.get_stories(file, unify_supporting, only_supporting, max_supporting)
            elif "test" in file:
                if test_task == "all":
                    self.test_stories += self.get_stories(file, unify_supporting, only_supporting, max_supporting)
                if test_task == "separate":
                    self.test_stories.append(self.get_stories(file, unify_supporting, only_supporting, max_supporting))
                elif "qa{}_".format(test_task) in file:
                    self.test_stories = self.get_stories(file, unify_supporting, only_supporting, max_supporting)
            elif "valid" in file:
                if valid_task == "all":
                    self.valid_stories += self.get_stories(file, unify_supporting, only_supporting, max_supporting)
                elif "qa{}_".format(valid_task) in file:
                    self.valid_stories = self.get_stories(file, unify_supporting, only_supporting, max_supporting)

        # Set data metrics
        all_data = self.train_stories + self.test_stories + self.valid_stories if test_task != "separate" \
            else self.train_stories + [j for i in self.test_stories for j in i] + self.valid_stories

        self.supports_max_num = max([len(story) for story, _, _ in all_data])
        self.support_max_len = max(map(len, (supporting for story, _, _ in all_data for supporting in story)))
        self.question_max_len = max(map(len, (question for _, question, _ in all_data)))

        self.vocab = set()
        for story, question, answer in all_data:
            self.vocab |= set([word for supporting in story for word in supporting] + question + [answer])
        self.vocab = sorted(self.vocab)

        # Reserve 0 for masking via pad_sequences TODO: separate vocab for questions and supports
        self.vocab_size = len(self.vocab) + 1

        print('-')
        print('Vocab size:', self.vocab_size, 'unique words')
        print('Supporting max number:', self.supports_max_num, 'sentences')
        print('Supporting max length:', self.support_max_len, 'words')
        print('Question max length:', self.question_max_len, 'words')
        print('Number of training stories:', len(self.train_stories))
        print('Number of validation stories:', len(self.valid_stories))
        print('Number of test stories:', len(self.test_stories if test_task != "separate" else self.test_stories[0]))
        print('-')
        print('Here\'s what a "story" tuple looks like (supporting, question, answer):')
        print(self.train_stories[5])
        print('-')
        print('Vectorizing the word sequences...')

        def one_hot(x, depth):
            new = np.zeros(depth)
            new[x] = 1
            return new

        self.dictionary = {word: i + 1 for i, word in enumerate(self.vocab)}
        # TODO: Remember to make embedding lookup table id 0 all 0s (padding) with no biases, or just mask

        def vectorize_and_pad(data):
            stories, questions, answers = [], [], []
            for s, q, a in data:
                s = [dict(supporting=[self.dictionary[word] for word in supporting],
                          supporting_len=len([self.dictionary[word] for word in supporting]))
                     for supporting in s]
                q = [self.dictionary[w] for w in q]

                stories.append(dict(story=s, supporting_num=len(s)))
                questions.append(dict(question=q, question_len=len(q)))
                # answers.append(dict(answer=one_hot(self.dictionary[a], self.vocab_size)))
                answers.append(dict(answer=self.dictionary[a]))

            # Return numpy arrays:
            # story [num_stories x supporting_max_num x supporting_max_len x vocab_size]
            # supporting_num [num_stories]
            # supporting_len [num_stories x supporting_max_num]
            # question [num_stories x question_max_len x vocab_size]
            # question_len [num_stories]
            # answer [num_stories x vocab_size]

            num_stories = len(data)

            # story = np.zeros((num_stories, self.supporting_max_num, self.supporting_max_len, self.vocab_size))
            # supporting_num = np.zeros(num_stories)
            # supporting_len = np.zeros((num_stories, self.supporting_max_num))
            # question = np.zeros((num_stories, self.question_max_len, self.vocab_size))
            # question_len = np.zeros(num_stories)
            # answer = np.zeros(num_stories, self.vocab_size)

            story = np.zeros((num_stories, self.supports_max_num, self.support_max_len))
            supporting_num = np.zeros(num_stories, np.int32)
            supporting_len = np.zeros((num_stories, self.supports_max_num), np.int32)
            supporting_pos = np.zeros((num_stories, self.supports_max_num), np.int32)
            question = np.zeros((num_stories, self.question_max_len))
            question_len = np.zeros(num_stories, np.int32)
            # answer = np.zeros((num_stories, self.vocab_size)) # Note that answer is still a one-hot while others aren't
            answer = np.zeros(num_stories)

            for st in range(num_stories):
                supporting_num[st] = stories[st]["supporting_num"]
                supporting_pos[st, :supporting_num[st]] = range(supporting_num[st])
                for su in range(supporting_num[st]):
                    supporting_len[st, su] = stories[st]["story"][su]["supporting_len"]
                    story[st, su, :supporting_len[st, su]] = stories[st]["story"][su]["supporting"]
                question_len[st] = questions[st]["question_len"]
                question[st, :question_len[st]] = questions[st]["question"]
                answer[st] = answers[st]["answer"]

            return {"supports": story, "support_num": supporting_num, "support_pos": supporting_pos,
                    "support_len": supporting_len, "question": question, "question_len": question_len,
                    "answer": answer}

        assert not unify_supporting
        self.train_stories = vectorize_and_pad(self.train_stories)
        self.test_stories = vectorize_and_pad(self.test_stories) if test_task != "separate" \
            else [vectorize_and_pad(ts) for ts in self.test_stories]
        self.valid_stories = vectorize_and_pad(self.valid_stories)

        self.train_stories_length = self.train_stories["supports"].shape[0]
        self.valid_stories_length = self.valid_stories["supports"].shape[0]
        self.test_stories_length = self.test_stories["supports"].shape[0] if test_task != "separate" \
            else self.test_stories[0]["supports"].shape[0]

        print('-')
        print('note that each word is a unique integer except answer, which is a corresponding one-hot.')
        print('-')
        print('supporting: integer tensor of shape (num_stories, supporting_max_num, supporting_max_length)')
        print('train shape:', self.train_stories["supports"].shape)
        print('valid shape:', self.valid_stories["supports"].shape)
        print('test shape:', self.test_stories["supports"].shape if test_task != "separate" else self.test_stories[0]["supports"].shape)
        print('-')
        print('questions: integer tensor of shape (num_stories, question_max_length)')
        print('train shape:', self.train_stories["question"].shape)
        print('valid shape:', self.valid_stories["question"].shape)
        print('test shape:', self.test_stories["question"].shape if test_task != "separate" else self.test_stories[0]["question"].shape)
        print('-')
        print('answers: one-hot tensor of shape (num_stories, vocab_size)')
        print('train shape:', self.train_stories["answer"].shape)
        print('valid shape:', self.valid_stories["answer"].shape)
        print('test shape:', self.test_stories["answer"].shape if test_task != "separate" else self.test_stories[0]["answer"].shape)
        print('-')
        print('Compiling...')
