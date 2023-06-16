from PIL import Image
from PIL.TiffTags import TAGS_V2
import sys
from pathlib import Path
from natsort import natsorted
import zarr
import numpy as np
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None


def readImage(filepath):
    image = Image.open(filepath)
    meta_data = {TAGS_V2[key].name: image.tag_v2[key]
                 for key in image.tag_v2}
    return image, meta_data


if __name__ == "__main__":

    assert len(sys.argv) == 4, "Needs to have 3 Arguments"

    # Automatically read all the files
    source_directory = Path(sys.argv[1])

    # Create a zarr file
    target_directory = Path(sys.argv[2])

    c0 = []
    c1 = []
    c2 = []
    c3 = []

    path = Path(sys.argv[3])

    filenames = [file.name for file in path.iterdir() if file.is_file()]

    for name in natsorted(filenames):
        if name[-6:-4] == "00":
            c0.append(name)
        elif name[-6:-4] == "01":
            c1.append(name)
        elif name[-6:-4] == "02":
            c2.append(name)
        else:
            c3.append(name)

    print("creating zarr datatset")
    zfile = zarr.open(target_directory/"livseg_data.zarr", mode="a")

    # Get the first file
    print("obtaining meta data")
    filenames = list(source_directory.iterdir())
    image, meta = readImage(filenames[0])

    shape = (len(filenames), meta["ImageWidth"], meta["ImageLength"])
    dtype = np.asarray(image).dtype

    dapi_data = zfile.require_dataset(
        "dapi", shape=shape, chunks=(1, 128, 128), dtype=dtype)
    channel_zero = zfile.require_dataset(
        "c0", shape=shape, chunks=(1, 128, 128), dtype=dtype)
    channel_one = zfile.require_dataset(
        "c1", shape=shape, chunks=(1, 128, 128), dtype=dtype)
    channel_two = zfile.require_dataset(
        "c2", shape=shape, chunks=(1, 128, 128), dtype=dtype)
    channel_three = zfile.require_dataset(
        "c3", shape=shape, chunks=(1, 128, 128), dtype=dtype)

    for key, value in meta.items():
        dapi_data.attrs[key] = value
        channel_zero.attrs[key] = value
        channel_one.attrs[key] = value
        channel_two.attrs[key] = value
        channel_three.attrs[key] = value

    print("reading/writing files")

    for channel, dataset in zip([filenames, c0, c1, c2, c3], [dapi_data, channel_zero, channel_one, channel_two, channel_three]):

        for z, name in tqdm(enumerate(natsorted(channel)), total=len(channel)):
            # Read the data in the file
            image, _ = readImage(name)

            # Save the data in the zarr dataset
            dataset[z] = np.asarray(image).T
