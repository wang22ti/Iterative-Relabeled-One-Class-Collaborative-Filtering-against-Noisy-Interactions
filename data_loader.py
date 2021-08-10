from torch.utils.data import Dataset
import os
import numpy as np
import torch
import copy


class TrainData(Dataset):
    def __init__(self, dataset, pos_rate=0.1, neg_rate=0.1):
        self.dataset = dataset
        train_path = os.path.join('data', '%s_train_ratings.lsvm' % self.dataset)
        self.num_user, self.num_item = 0, 0
        self.train_items = list()  # for interaction
        self.train_pairs = list()  # for interaction
        self.train_pos_items = list()  # for preference, ground-truth
        self.train_neg_items = list()  # for preference, ground-truth

        with open(train_path, 'r') as f:
            for user, line in enumerate(f):
                train_item, train_pos_item, train_neg_item = list(), list(), list()
                for item_rating in line.strip().split(' '):
                    item = int(item_rating.split(':')[0])
                    self.train_pairs.append((user, item))
                    train_item.append(item)

                    rating = float(item_rating.split(':')[1])
                    if rating >= 4:
                        train_pos_item.append(item)
                    else:
                        train_neg_item.append(item)

                self.train_items.append(train_item)
                self.train_pos_items.append(train_pos_item)
                self.train_neg_items.append(train_neg_item)

                self.num_item = max(self.num_item, max(train_item))
                self.num_user = max(self.num_user, user)

        self.num_user += 1
        self.num_item += 1
        self.all_users = [_ for _ in range(self.num_user)]

        self.raw_train_items = copy.deepcopy(self.train_items)
        self.pos_k = max(int(len(self.train_items[0]) * pos_rate), 1)
        self.neg_k = max(int(len(self.train_items[0]) * neg_rate), 1)
        self.interaction_mask = torch.zeros([self.num_user, self.num_item])
        for user in range(self.num_user):
            self.interaction_mask[user].scatter_(dim=0, index=torch.tensor(self.raw_train_items[user]), value=1.0)
        self.unobserved_mask = torch.ones([self.num_user, self.num_item]) - self.interaction_mask

        print('num_user:', self.num_user, 'num_item:', self.num_item)
        self.print_train_pos_rate()

    def __getitem__(self, idx):
        user = self.train_pairs[idx][0]
        pos_item = self.train_pairs[idx][1]
        neg_item = np.random.randint(self.num_item)
        while neg_item in self.train_items[user]:
            neg_item = np.random.randint(self.num_item)
        return user, pos_item, int(neg_item)

    def __len__(self):
        return len(self.train_pairs)

    def print_train_pos_rate(self):
        pos_rate = 0
        recall = 0
        for user in range(self.num_user):
            num_pos, num_total = 0, 0
            for item in self.train_items[user]:
                if item in self.train_pos_items[user]:
                    num_pos += 1
                    num_total += 1
                elif item in self.train_neg_items[user]:
                    num_total += 1
            if num_total == 0:
                continue
            pos_rate += num_pos / num_total
            if len(self.train_pos_items[user]) > 0:
                recall += num_pos / len(self.train_pos_items[user])

        pos_rate /= self.num_user
        recall /= self.num_user
        print('pos_rate:', pos_rate, 'recall', recall, 'f1', (pos_rate + recall) / 2)

    def generate_new_train_data(self, user_embedding, item_embedding, model_name='bpr', pos_rate=0.1, neg_rate=0.1,
                                plus=0, C=1.0, lambd=1):
        if model_name == 'bpr':
            user_embedding = user_embedding.detach()
            item_embedding = item_embedding.detach()
        self.train_pairs = list()

        all_ratings = np.array(torch.sigmoid(torch.mm(user_embedding, item_embedding.t())).cpu().tolist())
        item_embeddings = np.array(item_embedding.cpu().tolist())
        num_pos_change, num_true_pos_change, num_false_pos_change, num_neg_change, num_true_neg_change = 0, 0, 0, 0, 0
        for user in range(self.num_user):
            raw_train_items = self.train_pos_items[user] + self.train_neg_items[user]
            pos_k = max(int(len(raw_train_items) * pos_rate), 1)
            neg_k = max(int(len(raw_train_items) * neg_rate), 1)
            interaction_ratings, other_ratings = list(), list()
            for item in range(self.num_item):
                if item in raw_train_items:
                    interaction_ratings.append([item, all_ratings[user][item]])
                else:
                    other_ratings.append([item, all_ratings[user][item]])

            sorted_ratings = sorted(interaction_ratings, key=lambda x: x[1], reverse=True)
            pos_items = [_[0] for _ in sorted_ratings[:pos_k]]
            neg_items = [_[0] for _ in sorted_ratings[-neg_k:]]
            pos_items = self.easy_classify(user, pos_items, neg_items, item_embeddings, lambd)

            changed_items = list()
            for item in self.train_items[user]:
                if item not in pos_items:
                    changed_items.append(item)
                    if item in self.train_neg_items[user]:
                        num_true_pos_change += 1
                    elif item in self.train_pos_items[user]:
                        num_false_pos_change += 1
            num_pos_change += len(changed_items)

            sorted_other_ratings = sorted(other_ratings, key=lambda x: x[1], reverse=True)
            for (item, score) in [_ for _ in sorted_other_ratings[:plus]]:
                pos_items.append(item)
                num_neg_change += 1
                if item in self.train_pos_items[user]:
                    num_true_neg_change += 1

            self.train_items[user] = pos_items
            for item in pos_items:
                self.train_pairs.append((user, item))
        print('pos_change', num_pos_change, 'true_pos_change', num_true_pos_change, 'false_pos_change',
              num_false_pos_change, 'neg_change', num_neg_change, 'true_neg_change', num_true_neg_change, 'total', num_pos_change + num_neg_change)

    def save_train_data(self, epoch=0):
        with open(os.path.join('data', '%s_new_pos_%d.txt' % (self.dataset, epoch)), 'w') as f:
            for user in range(self.num_user):
                for item in self.train_items[user]:
                    print(item, end=' ', file=f)
                print(file=f)

    def easy_classify(self, user, pos_items, neg_items, item_embeddings, lambd):
        dim = item_embeddings.shape[1]
        pos_avg_feature, neg_avg_feature = np.zeros(shape=(dim,)), np.zeros(shape=(dim,))
        for item in pos_items:
            pos_avg_feature += item_embeddings[item]
        pos_avg_feature /= len(pos_items)

        for item in neg_items:
            neg_avg_feature += item_embeddings[item]
        neg_avg_feature /= len(neg_items)

        for item in self.train_items[user]:
            if item not in pos_items + neg_items:
                pos_dis = np.linalg.norm(item_embeddings[item] - pos_avg_feature, ord=2)
                neg_dis = np.linalg.norm(item_embeddings[item] - neg_avg_feature, ord=2)
                if pos_dis * lambd < neg_dis:
                    pos_items.append(item)
        return pos_items


class TestData:
    def __init__(self, train_data: TrainData):
        test_path = os.path.join('data', '%s_test_ratings.lsvm' % train_data.dataset)
        self.num_user, self.num_item = train_data.num_user, train_data.num_item
        self.trained_items = train_data.train_items

        self.test_items = list()  # for interaction
        self.test_pos_items = list()  # for preference
        self.test_neg_items = list()  # for preference
        num_pos, num_neg = 0, 0
        with open(test_path, 'r') as f:
            for user, line in enumerate(f):
                test_item, test_pos_item, test_neg_item = list(), list(), list()
                for item_rating in line.strip().split(' '):
                    item = int(item_rating.split(':')[0])
                    if item > self.num_item:
                        continue
                    test_item.append(item)
                    rating = float(item_rating.split(':')[1])
                    if rating >= 4:
                        test_pos_item.append(item)
                    else:
                        test_neg_item.append(item)

                num_pos += len(test_pos_item)
                num_neg += len(test_neg_item)
                self.test_items.append(test_item)
                self.test_pos_items.append(test_pos_item)
                self.test_neg_items.append(test_neg_item)
        print('num_test:', num_neg + num_pos, 'num_pos_test:', num_pos, 'num_neg_test:', num_neg)


class TestPreferenceData:
    def __init__(self, test_data: TestData):
        self.num_user = test_data.num_user
        self.num_item = test_data.num_item
        self.test_items = test_data.test_items
        self.test_pos_items = test_data.test_pos_items


class TestInteractionData:
    def __init__(self, test_data: TestData, args):
        self.test_items = list()
        self.num_user = test_data.num_user
        self.num_item = test_data.num_item
        self.test_pos_items = test_data.test_items
        self.all_items = set([_ for _ in range(self.num_item)])
        self.dataset = args['dataset']
        self.num_neg_for_test = args['num_neg_for_test']

        self.update_test_items(test_data.trained_items)

    def update_test_items(self, trained_items):
        self.test_items = list()
        for user in range(self.num_user):
            if self.dataset == 'netflix':
                test_items = [pos_item for pos_item in self.test_pos_items[user]]
                neg_items = list(self.all_items - set(trained_items[user]) - set(self.test_pos_items[user]))
                sampled_neg_items = [int(neg_item) for neg_item in
                                     np.random.choice(neg_items, self.num_neg_for_test, replace=False)]
                test_items += sampled_neg_items
            else:
                test_items = list(self.all_items - set(trained_items[user]))
            self.test_items.append(test_items)
