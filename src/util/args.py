import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description='Training for Skin Lesion Detector using ISIC2018 Task1 dataset')
    parser.add_argument('--model', choices=('ssd300', 'ssd512'), default='ssd300',
                        help='Model architecture (default: ssd300)')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--out', default='result',
                        help='Output directory')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Initialize the trainer from given file')
    parser.add_argument('--loaderjob', type=int, default=4,
                        help='Number of parallel data loading processes')

    return parser.parse_args()
