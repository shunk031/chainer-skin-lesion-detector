import xml.etree.ElementTree as ET

import chainer
import numpy as np
from chainercv.utils import read_image

from util import const


class ISIC2018Task1Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, img_fpaths, gt_fpaths):
        assert len(img_fpaths) == len(gt_fpaths), \
            f'# of image: {len(img_fpaths)} != # of ground truth: {len(gt_fpaths)}'
        self.annotations = self.load_annotations(img_fpaths, gt_fpaths)

    def load_annotations(self, img_fpaths, gt_fpaths):

        annotations = []
        for img_fpath, gt_fpath in zip(img_fpaths, gt_fpaths):
            anno_dict = self.parse_annotation(gt_fpath)
            annotations.append((img_fpath, anno_dict))

        return annotations

    def parse_annotation(self, xml_fpath):
        anno_dict = {'bbox': [], 'label': []}

        anno_xml = ET.parse(str(xml_fpath))
        for obj in anno_xml.findall('object'):
            bndbox = obj.find('bndbox')
            name = obj.find('name').text.strip()
            anno_dict['label'].append(const.LABELS.index(name))
            anno_dict['bbox'].append([
                int(bndbox.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

        return anno_dict

    def __len__(self):
        return len(self.annotations)

    def get_example(self, i):

        img_fpath, anno_dict = self.annotations[i]
        img = read_image(str(img_fpath), color=True)
        bbox = np.asarray(anno_dict['bbox'], dtype=np.float32)
        label = np.asarray(anno_dict['label'], dtype=np.float32)

        return img, bbox, label
