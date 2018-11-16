import xml.etree.ElementTree as ET
from xml.dom import minidom

from PIL import Image
from tqdm import tqdm

from util import const


def get_gt_fpaths():
    return [fpath for fpath in const.GT_DIR.iterdir() if fpath.suffix == '.png']


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


def get_bbox_from_image(gt_fpath):
    return Image.open(str(gt_fpath)).convert('RGB').getbbox()


def pretify_xml(elem):

    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)

    return reparsed.toprettyxml(indent='  ')


def main():

    gt_fpaths = get_gt_fpaths()
    for gt_fpath in tqdm(gt_fpaths):
        bbox = get_bbox_from_image()

        xml_file = make_voc_based_xml(
            folder_name=gt_fpath.parent.name,
            file_name=gt_fpath.name,
            bbox=bbox
        )

        xml_fpath = gt_fpath.parent / f'{gt_fpath.stem}.xml'
        save_voc_based_xml(xml_file, xml_fpath)


if __name__ == '__main__':
    main()
