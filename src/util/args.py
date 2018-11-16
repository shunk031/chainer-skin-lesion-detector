import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume', action='store_true', default=False)

    return parser.parse_args()
