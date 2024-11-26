# ref: https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/issues/8
# cmd: python pre_process_label.py src_path dataset output_path
from scipy.io import loadmat
import os
import argparse
from tqdm import tqdm
import json

def get_points(root_path, mat_path):
    m = loadmat(os.path.join(root_path, mat_path))
    return m['image_info'][0][0][0][0][0]

def get_image_list(root_path, sub_path):
    images_path = os.path.join(root_path, sub_path, 'images')
    images = [os.path.join(images_path, im) for im in
    os.listdir(os.path.join(root_path, images_path)) if 'jpg' in im]
    return images

def get_gt_from_image(image_path):
    gt_path = os.path.dirname(image_path.replace('images', 'ground-truth'))
    gt_filename = os.path.basename(image_path)
    gt_filename = 'GT_{}'.format(gt_filename.replace('jpg', 'mat'))
    return os.path.join(gt_path, gt_filename)

def ShanghaiTech(root_path, part_name, output_path):
    if part_name not in ['A', 'B']:
        raise NotImplementedError('Supplied dataset part does not exist')

    dataset_splits = ['train_data', 'val_data','test_data']
    for split in dataset_splits:
        part_folder = 'part_{}'.format(part_name)
        if part_name == 'A':
            sub_path = os.path.join(part_folder, '{}'.format(split))
        out_sub_path = os.path.join(part_folder, '{}'.format(split))

        images = get_image_list(root_path, sub_path=sub_path)
        try:
            os.makedirs(os.path.join(output_path, out_sub_path))
        except FileExistsError:
            print('Warning, output path already exists, overwriting')

        list_file = []
        for image_path in images:
            gt_path = get_gt_from_image(image_path)
            gt = get_points(root_path, gt_path)

            # for each image, generate a txt file with annotations
            new_labels_file = os.path.join(output_path, out_sub_path,
                                            os.path.basename(image_path).replace('jpg', 'txt'))
            with open(new_labels_file, 'w') as fp:
                for p in gt:
                    fp.write('{} {}\n'.format(p[0], p[1]))
            list_file.append((image_path, new_labels_file))

        # generate file with listing
        with open(os.path.join(output_path, part_folder,'{}.list'.format(split)), 'w') as fp:
            for item in list_file:
                fp.write('{} {}\n'.format(item[0], item[1]))


if __name__ == "__main__":
    root_path = r'C:\Users\User\CS331\Doan\CrowdCounting-P2PNet\DATA\ShanghaiTech'
    output_path = r'C:\Users\User\CS331\Doan\CrowdCounting-P2PNet\DATASET'
    ShanghaiTech(root_path, 'A', output_path)
