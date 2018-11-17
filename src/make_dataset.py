import argparse
import xml.etree.ElementTree as ET
from multiprocessing.pool import Pool
from xml.dom import minidom

from PIL import Image
from tqdm import tqdm

from util import const


def get_fpaths(data_dir, suffix):
    return [fpath for fpath in sorted(data_dir.iterdir(),
                                      key=lambda x: x.name) if fpath.suffix == suffix]


def make_voc_based_xml(folder_name, file_name, bbox):

    left, upper, right, lower = bbox
    annotation = ET.Element('annotation')

    annotation = ET.Element('annotation')
    tree = ET.ElementTree(element=annotation)
    folder = ET.SubElement(annotation, 'folder')
    filename = ET.SubElement(annotation, 'filename')
    objects = ET.SubElement(annotation, 'object')
    name = ET.SubElement(objects, 'name')
    pose = ET.SubElement(objects, 'pose')
    truncated = ET.SubElement(objects, 'truncated')
    difficult = ET.SubElement(objects, 'difficult')
    bndbox = ET.SubElement(objects, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    ymin = ET.SubElement(bndbox, 'ymin')
    xmax = ET.SubElement(bndbox, 'xmax')
    ymax = ET.SubElement(bndbox, 'ymax')

    folder.text = folder_name
    filename.text = file_name
    name.text = 'lesion'
    pose.text = 'frontal'
    truncated.text = '1'
    difficult.text = '0'
    xmin.text = str(left)
    ymin.text = str(upper)
    xmax.text = str(right)
    ymax.text = str(lower)

    return annotation


def save_voc_based_xml(xml_file, xml_fpath):

    xml_file = pretify_xml(xml_file)
    with xml_fpath.open('w') as wf:
        wf.write(xml_file)


def load_image(img_fpath):
    return Image.open(str(img_fpath))


def get_bbox_from_gt(gt):
    return gt.convert('RGB').getbbox()


def pretify_xml(elem):

    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)

    return reparsed.toprettyxml(indent='  ')


def preprocess_image_and_gt(img_fpath, gt_fpath):

    # rescale image
    img = load_image(img_fpath)
    img.thumbnail((const.MAX_SIZE, const.MAX_SIZE), Image.ANTIALIAS)
    img.save(str(const.PREPROCESSED_TRAIN_DIR / img_fpath.name))

    # rescale ground truth
    gt = load_image(gt_fpath)
    gt.thumbnail((const.MAX_SIZE, const.MAX_SIZE), Image.ANTIALIAS)
    gt.save(str(const.PREPROCESSED_GT_DIR / gt_fpath.name))

    # get bounding box from ground truth
    bbox = get_bbox_from_gt(gt)
    xml_file = make_voc_based_xml(
        folder_name=gt_fpath.parent.name,
        file_name=gt_fpath.name,
        bbox=bbox,
    )
    xml_fpath = const.PREPROCESSED_GT_DIR / f'{gt_fpath.stem}.xml'
    save_voc_based_xml(xml_file, xml_fpath)


def wapper_preprocess_image_and_gt(args):
    img_fpath, gt_fpath = args
    preprocess_image_and_gt(img_fpath, gt_fpath)


def main():

    parser = argparse.ArgumentParser(description='make dataset for training')
    parser.add_argument('--loaderjob', type=int, default=2)
    args = parser.parse_args()

    img_fpaths = get_fpaths(const.TRAIN_DIR, suffix='.jpg')
    gt_fpaths = get_fpaths(const.GT_DIR, suffix='.png')

    with Pool(processes=args.loaderjob) as pool:
        with tqdm(total=len(img_fpaths)) as pbar:
            for _ in tqdm(pool.imap(wapper_preprocess_image_and_gt,
                                    zip(img_fpaths, gt_fpaths))):
                pbar.update()


if __name__ == '__main__':
    main()
