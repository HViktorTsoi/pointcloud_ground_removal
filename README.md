# Ground removal for point cloud

## Install
1. First, clone this project，notice the SUBMODULE should also be cloned.
```bash
git clone --recursive https://github.com/HViktorTsoi/pointcloud_ground_removal.git
```

2. Compile and install

```bash
python setup.py install
```

## Usage
For details, please ref scripts/example.py
```python
import ground_removal_ext

pc = np.empty((25000, 4)) # x,y,z,intensity

# ground removal
segmentation = ground_removal_ext.ground_removal_kernel(pc, 0.2, 200) # distance_th=0.2, iter=200

print(segmentation.shape) # (25000, 5), the last channel represents if this point is ground,
# 0: ground, 255: Non-ground
```