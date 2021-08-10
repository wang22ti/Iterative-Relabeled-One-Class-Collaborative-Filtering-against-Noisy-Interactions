# Iterative-Relabeled-One-Class-Collaborative-Filtering-against-Noisy-Interactions

This is a Pytorch implementation of the model described in our paper:

>Z. Wang, Q. Xu, Z. Yang, X. Cao and Q. Huang. Implicit Feedbacks are Not Always Favorable: Iterative Relabeled One-Class Collaborative Filtering against Noisy Interactions. MM2021.

## Dependencies
- Pytorch >= 1.5.1
- numpy

## Data
We convert the datasets `ML100K`, `ML1M` and `Netflix` to our train and test files in the `data/` folder. 

To generate implicit training data, we randomly select $n_r$ ratings for each user as observed interactions, no matter whether the user likes the item or not. Meanwhile, we binarize the rest ratings according to a threshold of 4 to evaluate the performance in predicting true user preference, which is unavailable in implicit datasets. Note that users with less than $n_r$ ratings and items not appearing in the training set are filtered out.

The ratings are stored in the files *.lsvm. The data format of the line *user_id* is *item_id:rating*. The numeric ratings range from 1 to 5.

## Train

Here is an example to generate the new data.
```bash
python main.py
```

## Citation
Please cite our paper if you use this code in your own work:

```
@inproceedings{wang2021Iterative,
  title={Implicit Feedbacks are Not Always Favorable: Iterative Relabeled One-Class Collaborative Filtering against Noisy Interactions},
  author={Wang, Zitai and Xu, Qianqian and Yang, Zhiyong and Cao, Xiaochun and Huang, Qingming},
  booktitle={ACM on Multimedia Conference},
  year={2021}
}
```