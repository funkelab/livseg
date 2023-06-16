from PIL import Image
from PIL.TiffTags import TAGS_V2
from imageio import imread
import sys
from pathlib import Path
from natsort import natsorted
import zarr
import numpy as np
from tqdm import tqdm
from time import perf_counter
import dask.array as da


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

    filenames = [filename for filename in path.iterdir() if filename.is_file()]

    for filename in natsorted(filenames):
        channel = filename.name[-6:-4]
        if channel == "00":
            c0.append(filename)
        elif channel == "01":
            c1.append(filename)
        elif channel == "02":
            c2.append(filename)
        else:
            c3.append(filename)

    print("creating zarr datatset")
    zfile = zarr.open(target_directory/"livseg_data.zarr", mode="a")

    # Get the first file
    print("reading/writing files")

    for channel, dsname in zip(
            [filenames, c0, c1, c2, c3],
            ["dapi", "c0", "c1", "c2", "c3"]
    ):

        print("obtaining meta data")
        image, meta = readImage(channel[0])

        images = []
        for z, name in tqdm(enumerate(natsorted(channel)),
                            total=len(channel), desc=dsname):
            # Read the data in the file
            image = imread(name)
            images.append(image.T)
        full_stack = np.stack(images)
        # Save the data in the zarr dataset
        array = da.from_array(full_stack, chunks=(32, 32, 32))
        t0 = perf_counter()
        array.to_zarr(target_directory / "livseg_data.zarr", dsname,
                      overwrite=True)
        print(f"Dataset {dsname} took {perf_counter() - t0}s to save")
        # Add metadata
        for key, value in meta.items():
            # TODO check with Caleb which attributes he needs
            zfile[dsname].attrs[key] = value
