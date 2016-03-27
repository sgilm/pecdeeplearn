import os
import nibabel
import tools


def read_data_path():
    directory = '.'
    directory_list = next(os.walk(directory))[1]
    while 'data' not in directory_list:
        if 'pecdeeplearn' in directory_list:
            return -1
        directory += '/..'
        directory_list = next(os.walk(directory))[1]

    with open(os.path.join(directory, 'data', 'path.txt'), 'r') as f:
        path = f.readline().strip()

    return path


def list_volumes():
    data_path = read_data_path()
    hdr_volumes = []
    img_volumes = []
    for filename in os.listdir(os.path.join(data_path, 'mris')):
        if '.hdr' in filename:
            hdr_volumes.append(filename.split('.')[0])
        elif '.img' in filename:
            img_volumes.append(filename.split('.')[0])

    mri_volumes = [volume for volume in hdr_volumes if volume in img_volumes]

    seg_volumes = []
    for filename in os.listdir(os.path.join(data_path, 'segmentations')):
        volume = filename.split('.')[0].replace('segpec_', '')
        seg_volumes.append(volume)

    volumes = [volume for volume in seg_volumes if volume in mri_volumes]

    return volumes


def load_volume(volume_name):
    data_path = read_data_path()
    for filename in os.listdir(os.path.join(data_path, 'mris')):
        if volume_name in filename and '.hdr' in filename:
            mri_filename = filename

    for filename in os.listdir(os.path.join(data_path, 'segmentations')):
        if volume_name in filename:
            seg_filename = filename
    
    mri = nibabel.load(os.path.join(data_path, 'mris', mri_filename))
    seg =  nibabel.load(os.path.join(data_path, 'segmentations', seg_filename))
    volume = tools.Volume(mri.get_data(), seg.get_data())

    return volume


if __name__ == '__main__':
    volumes = list_volumes()
    volume = load_volume(volumes[0])