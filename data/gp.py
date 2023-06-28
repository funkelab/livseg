import gunpowder as gp
import matplotlib.pyplot as plt

raw_data = "/Volumes/funkelab/livseg/data/crops/lobule1_central_crop.zarr"
label_data = "/Volumes/funkelab/livseg/data/test.n5"

voxel_size = (198, 45, 45)
voxel_size = gp.Coordinate(voxel_size)
shape = (64, 64, 64)
size = gp.Coordinate(shape)

raw = gp.ArrayKey('RAW')
label = gp.ArrayKey('LABEL')

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

pipeline = combined_source

request = gp.BatchRequest()

request[raw] = gp.Roi((0, 0, 0), size*voxel_size)
request[label] = gp.Roi((0, 0, 0), size*voxel_size)

with gp.build(pipeline):
    batch = pipeline.request_batch(request)

print(f"batch returned: {batch}")
plt.imshow(batch[raw].data[0])
plt.show()
plt.imshow(batch[label].data[0])
