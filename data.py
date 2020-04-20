from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import torch
from torch.utils.data.sampler import WeightedRandomSampler


def get_annotation(path):
    texts = path.read_text().split('\n')
    columns = np.array(texts[1].split(" "))
    columns = columns[columns != ""]

    df = []
    for txt in texts[2:]:
        txt = np.array(txt.split(" "))
        txt = txt[txt != ""]
        df.append(txt)

    df = pd.DataFrame(df)
    columns = ["image_id"] + list(columns)
    df.columns = columns
    df = df.dropna()

    ## cast to integer
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = (pd.to_numeric(df[nm], downcast="integer") + 1) / 2
    return df


def train_val_test_stratified_split(df, n_folds, random_state=24):
    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=random_state)

    val_ind, test_ind, train_ind = [], [], []
    for i, fold in enumerate(k_fold.split(df.path, df.label)):
        if i == 0:
            val_ind = fold[1]
        elif i == 1:
            test_ind = fold[1]
        else:
            train_ind.append(fold[1])
    train_ind = np.hstack(train_ind)

    return df.iloc[train_ind].reset_index(drop=True), \
           df.iloc[val_ind].reset_index(drop=True), \
           df.iloc[test_ind].reset_index(drop=True)


def load_celeba_train_val_test(data_root, target_column, n_folds,
                               downsample_bigger_class=None):
    data_root = Path(data_root)
    dataset_folder = data_root / 'img_align_celeba'
    list_attr = data_root / 'list_attr_celeba.txt'

    attrs = get_annotation(list_attr)
    attrs['path'] = attrs.image_id.apply(lambda x: str(dataset_folder / x))

    df = attrs[['path', target_column]]
    df.columns = ['path', 'label']

    if downsample_bigger_class is not None:
        print('Before downsampling')
        print(df.label.value_counts())
        bigger_label = df.label.value_counts().index[0]
        df = pd.concat((df[df.label == bigger_label].sample(
            frac=downsample_bigger_class), df[df.label != bigger_label]))
        print('After downsampling')
        print(df.label.value_counts())

    train_df, val_df, test_df = train_val_test_stratified_split(df, n_folds)
    return train_df, val_df, test_df


def class_imbalance_sampler(labels):
    labels = labels.astype(int).values
    class_count = torch.bincount(torch.from_numpy(labels)).float()
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]

    sampler = WeightedRandomSampler(sample_weights, len(labels))
    return sampler


class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_transforms):
        self.data = data
        self.image_transforms = image_transforms

    def __getitem__(self, index):
        path = self.data.path[index]
        img = Image.open(path)
        img = self.image_transforms(img)
        label = int(self.data.label[index])
        return img, label

    def __len__(self):
        return len(self.data)


IMAGENET_STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
