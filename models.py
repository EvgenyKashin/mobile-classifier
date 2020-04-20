import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import timm
import data
from data import PandasDataset


class MobileModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        if self.hparams.model_architecture == 'squeezenet1_1':
            self.model = torchvision.models.squeezenet1_1(
                pretrained=self.hparams.pretrained)
            self.model.classifier[1] = nn.Conv2d(512, self.hparams.num_classes,
                                            kernel_size=1)
        elif self.hparams.model_architecture == 'mobilenet_v2':
            self.model = torchvision.models.mobilenet_v2(
                pretrained=self.hparams.pretrained
            )
            self.model.classifier[1] = nn.Linear(1280, self.hparams.num_classes)
        elif self.hparams.model_architecture == 'mnasnet0_5':
            self.model = torchvision.models.mnasnet0_5(
                pretrained=self.hparams.pretrained
            )
            self.model.classifier[1] = nn.Linear(1280, self.hparams.num_classes)
        elif self.hparams.model_architecture == 'shufflenet_v2_x1_0':
            self.model = torchvision.models.shufflenet_v2_x1_0(
                pretrained=self.hparams.pretrained
            )
            self.model.fc = nn.Linear(1024, self.hparams.num_classes)
        elif self.hparams.model_architecture.startswith('tf_'):
            self.model = timm.create_model(self.hparams.model_architecture,
                                           num_classes=self.hparams.num_classes,
                                           pretrained=self.hparams.pretrained)
        else:
            raise ValueError(f'Unknown model architecture:'
                             f' {self.hparams.model_architecture}')

    def forward(self, x):
        x = self.model(x)
        return x

    def prepare_data(self):
        print('Dataset loading..')
        train_df, val_df, test_df = data.load_celeba_train_val_test(
            self.hparams.data_root, self.hparams.target_column,
            self.hparams.split_folds, self.hparams.downsample_bigger_class)

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomAffine(degrees=(0, 0), translate=(0.05, 0.05),
                                        scale=(0.9, 1.1)),
                transforms.RandomChoice([
                    transforms.RandomAffine(degrees=(-20, 20)),
                    transforms.RandomAffine(degrees=(0, 0),
                                            shear=(-5, 5, -5, 5))
                ]),

                transforms.RandomChoice([
                    transforms.ColorJitter(hue=0.03, saturation=0.3),
                    transforms.ColorJitter(contrast=0.2, brightness=0.2),
                ]),

                transforms.RandomCrop(self.hparams.crop_size),
                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),
                transforms.Normalize(*data.IMAGENET_STATS)
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(self.hparams.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(*data.IMAGENET_STATS)
            ])
        }

        self.train_sampler = data.class_imbalance_sampler(train_df.label)
        self.val_sampler = data.class_imbalance_sampler(val_df.label)

        self.train_dataset = PandasDataset(train_df, data_transforms['train'])
        self.val_dataset = PandasDataset(val_df, data_transforms['val'])
        self.test_dataset = PandasDataset(test_df, data_transforms['val'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          sampler=self.val_sampler, )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        scheduler = ExponentialLR(optimizer, self.hparams.lr_gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def _accuracy_metric(outputs, y_true):
        _, preds = torch.max(outputs, 1)
        return torch.sum(preds == y_true.data).double() / len(y_true)

    def _step(self, batch, batch_idx, prefix='', with_logs=True):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self._accuracy_metric(logits, y)

        loss_name = prefix + 'loss'
        acc_name = prefix + 'acc'

        logs = {loss_name: loss, acc_name: acc}
        outputs = logs.copy()
        if with_logs:
            outputs['log'] = logs
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val_',
                          with_logs=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        _, preds = torch.max(logits, 1)
        outputs = {'y_true': y.cpu().numpy(), 'y_pred': preds.cpu().numpy()}
        return outputs

    def _epoch_end(self, outputs, prefix='', with_logs=True):
        loss_name = prefix + 'loss'
        acc_name = prefix + 'acc'

        avg_loss = torch.stack([x[loss_name] for x in outputs]).mean()
        avg_acc = torch.stack([x[acc_name] for x in outputs]).mean()
        logs = {loss_name: avg_loss, acc_name: avg_acc}
        outputs = logs.copy()
        if with_logs:
            outputs['log'] = logs
        return outputs

    def training_epoch_end(self, outputs):
        return self._epoch_end(outputs, with_logs=False)

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, 'val_')

    def test_epoch_end(self, outputs):
        y_true = np.hstack([x['y_true'] for x in outputs])
        y_pred = np.hstack([x['y_pred'] for x in outputs])
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        return {'accuracy': acc, 'f1_score': f1, 'roc_auc': roc_auc}
