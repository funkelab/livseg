from PIL import Image
from PIL.TiffTags import TAGS_V2
from imageio.v2 import imread
import sys
from pathlib import Path
from natsort import natsorted
import zarr
import numpy as np
from tqdm import tqdm
from time import perf_counter

Image.MAX_IMAGE_PIXELS = None


def read_image_meta_data(filepath):
    image = Image.open(filepath)
    meta_data = {TAGS_V2[key].name: image.tag_v2[key]
                 for key in image.tag_v2}
    return meta_data


def process_lobule(source_dir, channel_subdir, zarr_name, overwrite=False):
    """ Create a zarr to hold one lobule with 5 channels in datasets.
    Args:
        source_dir: Directory for the lobule. Has "DAPI" subdirectory and
            channel_subdir.
        channel_subdir: Name of the subdirectory with the channel tiffs
        zarr_name: output zarr name (full path)
    """
    # find size and dtype
    dapi_path = Path(source_dir, "DAPI")
    try:
        sample_file = next(dapi_path.glob("*.tif"))
    except:
        raise RuntimeError(f"No tiff files in {dapi_path}")
    sample_image = imread(sample_file)
    meta_data = read_image_meta_data(sample_file)
    image_shape = sample_image.shape  # one z slice
    # get number of z slices
    z_shape = len(list(dapi_path.glob("*.tif")))
    shape = (z_shape, *image_shape)
    print(f"{shape=}")
    data_type = sample_image.dtype
    print(f"{data_type=}")

    # make empty zarr with 5 groups of given size and dtype
    if Path(zarr_name).exists() and not overwrite:
        raise RuntimeError(f"Zarr {zarr_name} already exists")
    zarr_store = zarr.NestedDirectoryStore(zarr_name)
    base = zarr.open(zarr_store, 'w')
    chunk_shape = (32, 1024, 1024)

    # find number of channels in channel_subdir
    channel_path = Path(source_dir, channel_subdir)
    channels = set()
    for filename in channel_path.glob("*.tif"):
        channel = filename.name[-6:-4]
        channels.add(channel)
    channels = list(channels)
    datasets = ["DAPI", ] + channels
    
    for dataset in datasets:
        base.create_dataset(dataset, shape=shape, chunks=chunk_shape, dtype=data_type)
    
        # Add metadata
        base[dataset].attrs["resolution"] = (
            1,  # TODO: just a guess, need to replace with correct value later
            int(1e7/meta_data["YResolution"]),
            int(1e7/meta_data["XResolution"])
        )
        
        base[dataset].attrs["axis_names"] = ("z", "y", "x")
        base[dataset].attrs["units"] = ("nm", "nm", "nm")

    # loop over DAPI files, load, and save
    dapi_dataset = base['DAPI']
    for index, dapi_file in enumerate(tqdm(natsorted(dapi_path.glob("*.tif")))):
        image = imread(dapi_file)
        dapi_dataset[index] = image

    
    for index, channel_file in enumerate(tqdm(natsorted(channel_path.glob("*.tif")))):
        channel = channel_file.name[-6:-4]
        image = imread(channel_file)
        base[channel][index] = image

    

if __name__ == "__main__":
    source_dir = '/groups/feliciano/felicianolab/For_Alex_and_Mark/Male/CNT/Liv1/Lobule1'
    channel_subdir = 'Mito_Perox_LD_Actin'
    zarr_name = '/nrs/funke/data/confocal/feliciano_liver/male/cnt/liv1/lobule1.zarr'
    process_lobule(
        source_dir,
        channel_subdir,
        zarr_name,
        overwrite=True)