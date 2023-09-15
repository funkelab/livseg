import zarr

filepath = "/groups/funke/funkelab/livseg/data/livseg_data.zarr"
img = zarr.open(filepath, 'r')['c0']
target_directory = '/groups/funke/funkelab/livseg/data/crops/middle_of_acinus_crop2.zarr'

# TODO: HARDCODED!! need to be changed for different crops
center = (50, 16000, 6250)  # (z, x, y)
size = (64, 256, 256)

div_size = tuple(ti//2 for ti in size)

start_point = tuple(map(lambda i, j: int(i - j), center, div_size))
end_point = tuple(map(lambda i, j: int(i + j), center, div_size))

crop = img[start_point[0]:end_point[0],
           start_point[1]:end_point[1],
           start_point[2]:end_point[2]]

zarr_file = zarr.open(target_directory, 'w')

zarr_file['raw'] = crop

zarr_file['raw'].attrs['resolution'] = (198, 45, 45)
zarr_file['raw'].attrs['center_point'] = center
zarr_file['raw'].attrs['offset'] = start_point
