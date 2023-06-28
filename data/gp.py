import gunpowder as gp
import matplotlib.pyplot as plt
import numpy as np

#from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.torch import Train
from lsd.train.gp import AddLocalShapeDescriptor
#from tqdm import tqdm
from model import MtlsdModel, WeightedMSELoss
import torch

raw_data = "/Volumes/funkelab/livseg/data/crops/lobule1_central_crop.zarr"
label_data = "/Volumes/funkelab/livseg/data/test.n5"

voxel_size = (198, 45, 45)
voxel_size = gp.Coordinate(voxel_size)
shape = (64, 64, 64)
size = gp.Coordinate(shape)
output_size = size*voxel_size

raw = gp.ArrayKey('RAW')
label = gp.ArrayKey('LABEL')
gt_lsds = gp.ArrayKey('GT_LSDS')
lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
pred_lsds = gp.ArrayKey('PRED_LSDS')
gt_affs = gp.ArrayKey('GT_AFFS')
affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
pred_affs = gp.ArrayKey('PRED_AFFS')


raw_source = gp.ZarrSource(
    raw_data,
    {raw: 'c0'},
    {raw: gp.ArraySpec(interpolatable=True)}
)

label_source = gp.ZarrSource(
    label_data,
    {label: 'labels'},
    {label: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size)}
)

combined_source = (raw_source, label_source) + gp.MergeProvider()
normalize = gp.Normalize(raw)
random_location = gp.RandomLocation(
    min_masked=0.01,
    mask=label

)
simple_augment = gp.SimpleAugment(
    transpose_only=[1, 2]
)
intensity_augment = gp.IntensityAugment(
    raw,
    scale_min=0.9,
    scale_max=1.1,
    shift_min=-0.1,
    shift_max=0.1
)

pipeline = combined_source
pipeline += normalize
pipeline += random_location
pipeline += simple_augment
pipeline += intensity_augment
pipeline += gp.GrowBoundary(label)
pipeline += AddLocalShapeDescriptor(
    label,
    gt_lsds,
    lsds_mask=lsds_weights,
    sigma=10
)
pipeline += gp.AddAffinities(
    affinity_neighborhood=[
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]],
    labels=label,
    affinities=gt_affs,
    dtype=np.float32,
    affinities_mask=affs_weights
)

request = gp.BatchRequest()

request.add(raw, output_size)
request.add(label, output_size)
request.add(gt_lsds, output_size)
request.add(lsds_weights, output_size)
request.add(pred_lsds, output_size)
request.add(gt_affs, output_size)
request.add(affs_weights, output_size)
request.add(pred_affs, output_size)

# defining variables

in_channels = 1
num_fmaps = 16
fmap_inc_factor = 2
downsample_factors = [(2, 2, 2), (2, 2, 2)]
# kernel_size_down =
# kernel_size_up =
constant_upsample = True

lr = 1e-5

model = MtlsdModel(
    in_channels=in_channels,
    num_fmaps=num_fmaps,
    fmap_inc_factor=fmap_inc_factor,
    downsample_factors=downsample_factors,
    constant_upsample=constant_upsample
)

loss = WeightedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

pipeline += gp.Unsqueeze([raw])

train = Train(
    model,
    loss,
    optimizer,
    inputs={
        'input': raw
    },
    outputs={
        0: pred_lsds,
        1: pred_affs
    },
    loss_inputs={
        'lsds_prediction': pred_lsds,
        'lsds_target': gt_lsds,
        'lsds_weights': lsds_weights,
        'affs_prediction': pred_affs,
        'affs_target': gt_affs,
        'affs_weights': affs_weights
    }
)

pipeline += train

with gp.build(pipeline):
    batch = pipeline.request_batch(request)

print(f"batch returned: {batch}")

plt.imshow(batch[raw].data[0])
plt.show()
plt.imshow(batch[label].data[0])
