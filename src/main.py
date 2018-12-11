import cv2  # NOQA # isort:skip
cv2.setNumThreads(0)  # NOQA
import os

import chainer
from chainer import training
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer.training import extensions, triggers
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling

from models import ARCHS
from util import const
from util.args import parse_args
from util.cross_validation import load_train_test
from util.multi_box_train import MultiboxTrainChain
from util.resource import Resource
from util.skin_lesion_dataset import ISIC2018Task1Dataset
from util.transforms import Transform


def main():

    args = parse_args()
    res = Resource(args, train=True)

    train, test, train_gt, test_gt = load_train_test(
        train_dir=const.PREPROCESSED_TRAIN_DIR,
        gt_dir=const.XML_DIR)
    res.log_info(f'Train: {len(train)}, test: {len(test)}')

    model = ARCHS[args.model](n_fg_class=len(const.LABELS),
                              pretrained_model='imagenet')
    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    train_dataset = TransformDataset(
        ISIC2018Task1Dataset(train, train_gt),
        Transform(model.coder, model.insize, model.mean)
    )
    train_iter = chainer.iterators.MultithreadIterator(
        train_dataset, args.batchsize, n_threads=args.loaderjob)

    test_dataset = TransformDataset(
        ISIC2018Task1Dataset(test, test_gt),
        Transform(model.coder, model.insize, model.mean))
    test_iter = chainer.iterators.MultithreadIterator(
        test_dataset, args.batchsize, shuffle=False, repeat=False,
        n_threads=args.loaderjob)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)
    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=False,
            label_names=const.LABELS))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/loss', 'main/loss/loc', 'main/loss/conf',
        'validation/main/map']))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    snapshot_trigger = triggers.MaxValueTrigger(
        key='validation/main/map')
    snapshot_object_trigger = triggers.MaxValueTrigger(
        key='validation/main/map')
    trainer.extend(extensions.snapshot(filename='snapshot_best.npz'),
                   trigger=snapshot_trigger)
    trainer.extend(extensions.snapshot_object(model, 'model_best.npz'),
                   trigger=snapshot_object_trigger)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # save last model
    chainer.serializers.save_npz(
        os.path.join(args.out, 'snapshot_last.npz'), trainer)
    chainer.serializers.save_npz(
        os.path.join(args.out, 'model_last.npz'), model)


if __name__ == '__main__':
    main()
