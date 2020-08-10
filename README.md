# Ground removal for point cloud

## Install
1. 首先将项目clone下来，注意submodule也要clone
```bash
git clone --recursive https://github.com/HViktorTsoi/pointcloud_ground_removal.git
```

2. 编译并安装

```bash
python setup.py install
```

## 使用
详情参考scripts/example.py
```python
import ground_removal_ext

pc = np.empty((25000, 4)) # x,y,z,intensity

# ground removal
segmentation = ground_removal_ext.ground_removal_kernel(pc, 0.2, 200) # distance_th=0.2, iter=200

print(segmentation.shape) # (25000, 5), 最后一个维度代表是否为地面,0: 地面, 255: 非地面
```