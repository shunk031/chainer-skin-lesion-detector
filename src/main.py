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
        train_dir=const.TRAIN_DIR, gt_dir=const.GT_DIR)

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
    train_iter = chainer.iterators.MultiprocessIterator(train_dataset, args.batchsize)

    test_dataset = TransformDataset(
        ISIC2018Task1Dataset(test, test_gt),
        Transform(model.coder, model.insize, model.mean))
    test_iter = chainer.iterators.MultiprocessIterator(test_dataset, args.batchsize)

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))
    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=const.LABELS),
        trigger=(10000, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=(10000, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(120000, 'iteration'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
