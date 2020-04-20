from argparse import ArgumentParser
import torch
import torch.nn as nn
import apex
from pytorch_lightning import Trainer
from models import MobileModel


bool_t = lambda x: (str(x).lower() == 'true')


def fuse_conv_bn(model):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Sequential):
            for idx in range(len(mod)):
                if isinstance(mod[idx], nn.Conv2d) and isinstance(mod[idx + 1],
                                                       nn.BatchNorm2d):
                    torch.quantization.fuse_modules(mod, [str(idx),
                                                          str(idx + 1)],
                                                    inplace=True)
    print('Conv + bn fused')


def main(hparams):
    mobile_model = MobileModel.load_from_checkpoint(hparams.checkpoint_path)
    mobile_model.freeze()

    trainer = Trainer(gpus=hparams.gpus, amp_level=hparams.amp_level,
                      precision=16, logger=False)
    if hparams.fuse_conv_bn:
        assert mobile_model.hparams.model_architecture in [
            'mnasnet0_5', 'shufflenet_v2_x1_0'
            ]
        fuse_conv_bn(mobile_model)

    trainer.test(mobile_model)
    name = mobile_model.hparams.model_architecture

    if hparams.save_onnx:
        try:
            mobile_model = apex.amp.initialize(mobile_model.float().cuda(),
                                               opt_level='O3')
            stub_img = torch.ones((1, 3,hparams.fixed_image_size,
                                   hparams.fixed_image_size)).cuda().half()
            torch.onnx.export(mobile_model.model,
                              stub_img,
                              f'weights/{name}_fp16.onnx',
                              verbose=False,
                              do_constant_folding=True)
        except Exception as e:
            print('Onnx export error, try harder. Saved in pytorch format only')
            print(e)

    if hparams.save_pytorch:
        mobile_model = MobileModel.load_from_checkpoint(hparams.checkpoint_path)
        mobile_model.freeze()

        if hparams.fuse_conv_bn:
            assert mobile_model.hparams.model_architecture in [
                'mnasnet0_5', 'shufflenet_v2_x1_0'
            ]
            fuse_conv_bn(mobile_model)

        torch.save(mobile_model.model.half().cpu().state_dict(),
                   f'weights/{name}_fp16.pth')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--save_onnx', type=bool_t, default=True)
    parser.add_argument('--save_pytorch', type=bool_t, default=True)
    parser.add_argument('--fixed_image_size', type=int, default=160)
    parser.add_argument('--amp_level', type=str, default='O2')
    parser.add_argument('--fuse_conv_bn', type=bool_t, default=False)

    hparams = parser.parse_args()
    print(hparams)
    main(hparams)
