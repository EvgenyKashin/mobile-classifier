from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from facenet_pytorch import MTCNN
import data


def load_model(num_classes, model_weights):
    model = torchvision.models.shufflenet_v2_x1_0()
    model.fc = nn.Linear(1024, num_classes)
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    return model


def main(hparams):
    mtcnn = MTCNN(image_size=hparams.img_size, margin=hparams.crop_margin,
                  post_process=False)
    model = load_model(hparams.num_classes, hparams.model_weights)
    norm_transform = transforms.Normalize(*data.IMAGENET_STATS)

    folder_path = Path(hparams.folder_path)
    paths = sorted(list(folder_path.iterdir()))
    if hparams.predict_max_count is not None:
        paths = paths[:hparams.predict_max_count]

    for path in paths:
        try:
            img = Image.open(path)
        except Exception as e:
            print(f'Skip {path}')
            print(e)
            continue

        img_crop = mtcnn(img)
        if img_crop is None:
            print(f'Face not found in {path}')
            continue

        img_crop = norm_transform(img_crop / 255.)
        pr_crop = model(img_crop.unsqueeze(0))[0]
        pr_crop = torch.argmax(pr_crop).item()

        if pr_crop == 1:
            print(f'Eyeglasses are founded in {path}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--predict_max_count', type=int, default=None)
    parser.add_argument('--img_size', type=int, default=160)
    parser.add_argument('--crop_margin', type=int, default=40)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model_weights', type=str,
                        default='weights/shufflenet_v2_x1_0_fp16.pth')

    hparams = parser.parse_args()
    print(hparams)
    main(hparams)
