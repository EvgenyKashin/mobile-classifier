from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from models import MobileModel


bool_t = lambda x: (str(x).lower() == 'true')


def main(hparams):
    mobile_model = MobileModel(hparams)
    logger = TensorBoardLogger('lightning_logs',
                               name=hparams.model_architecture)
    trainer = Trainer(gpus=hparams.gpus, profiler=True, val_check_interval=0.5,
                      logger=logger, max_epochs=hparams.max_epochs,
                      amp_level=hparams.amp_level, precision=16)
    trainer.fit(mobile_model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data',
                        help='path where dataset(celeba) is stored')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--model_architecture', type=str,
                        default='shufflenet_v2_x1_0', choices=[
            'squeezenet1_1', 'mobilenet_v2', 'mnasnet0_5',
            'shufflenet_v2_x1_0', 'tf_efficientnet_lite0',
             'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100'])

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_gamma', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--pretrained', type=bool_t, default=True)
    parser.add_argument('--target_column', type=str, default='Eyeglasses')
    parser.add_argument('--split_folds', type=int, default=10)
    parser.add_argument('--downsample_bigger_class', type=int, default=0.25)
    parser.add_argument('--amp_level', type=str, default='O2')

    hparams = parser.parse_args()
    print(hparams)
    main(hparams)
