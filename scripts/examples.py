import array
import time

import numpy as np
import ground_removal_ext


def load_pc(bin_file_path):
    """
    load pointcloud file (KITTI format)
    :param bin_file_path:
    :return:
    """
    with open(bin_file_path, 'rb') as bin_file:
        pc = array.array('f')
        pc.frombytes(bin_file.read())
        pc = np.array(pc).reshape(-1, 4)
        return pc


if __name__ == '__main__':
    pc = load_pc(
        '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/KITTI/tracking/training/velodyne/0008/000100.bin')

    # ground removal
    tic = time.time()
    segmentation = ground_removal_ext.ground_removal_kernel(pc, 0.2, 200)
    toc = time.time()
    print('TIME used: ', toc - tic)

    # Nx5, x, y, z, intensity, is_ground
    print(segmentation.shape)
    print(segmentation)

    try:
        from mayavi import mlab

        segmentation[..., 3] = segmentation[..., 4]
        mlab.points3d(segmentation[:, 0], segmentation[:, 1], segmentation[:, 2], segmentation[:, 3], mode='point')
        mlab.show()
    except ImportError:
        print('mayavi not installed, skip visualization')
