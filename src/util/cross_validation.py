from sklearn.model_selection import train_test_split

from util import const


def load_dataset_fpaths(data_dir, suffix):
    return [fpath for fpath in sorted(data_dir.iterdir(),
                                      key=lambda x: x.name) if fpath.suffix == suffix]


def load_train_test(train_dir, gt_dir):
    img_fpaths = load_dataset_fpaths(train_dir, suffix='.jpg')
    gt_fpaths = load_dataset_fpaths(gt_dir, suffix='.xml')

    # import pdb
    # pdb.set_trace()

    assert len(img_fpaths) == len(gt_fpaths)

    return isic_task1_train_test_split(img_fpaths, gt_fpaths)


def isic_task1_train_test_split(*arrays):
    return train_test_split(*arrays, random_state=const.SEED,
                            test_size=const.VALIDATION_SIZE)
