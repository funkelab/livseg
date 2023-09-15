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


Image.MAX_IMAGE_PIXELS = None


def read_image_meta_data(filepath):
    image = Image.open(filepath)
    meta_data = {TAGS_V2[key].name: image.tag_v2[key]
                 for key in image.tag_v2}
    return meta_data


if __name__ == "__main__":

    print("Starting :)")

    assert len(sys.argv) == 4, "Needs to have 3 Arguments"

    # Automatically read all the files
    source_directory = Path(sys.argv[1])

    # Create a zarr file
    target_directory = Path(sys.argv[2])

    c0_filenames = []
    c1_filenames = []
    c2_filenames = []
    c3_filenames = []

    path = Path(sys.argv[3])

    print("Searching for channel files...")

    dapi_filenames = [filename for filename in source_directory.iterdir() if filename.is_file()]
    other_channel_filenames = [
        filename for filename in path.iterdir() if filename.is_file()]

    print("Separating other channels...")

    for filename in tqdm(natsorted(other_channel_filenames)):
        ch_id = filename.name[-6:-4]
        if ch_id == "00":
            c0_filenames.append(filename)
        elif ch_id == "01":
            c1_filenames.append(filename)
        elif ch_id == "02":
            c2_filenames.append(filename)
        else:
            c3_filenames.append(filename)

    print("creating zarr datatset")
    zarr_container = zarr.open(target_directory / "livseg_data.zarr", "a")

    # Get the first file
    print("reading/writing files")

    for channel_filenames, dsname in zip(
            [dapi_filenames, c0_filenames, c1_filenames, c2_filenames, c3_filenames],
            ["dapi", "c0", "c1", "c2", "c3"]
    ):

        print("obtaining meta data")
        meta_data = read_image_meta_data(channel_filenames[0])
        print(meta_data)

        print("Reading images...")
        filenames = natsorted(channel_filenames)
        images = [
            imread(filename).T
            for filename in tqdm(filenames, desc=dsname)
        ]

        # Save the data in the zarr dataset
        print("Saving to zarr...")
        zarr_container[dsname] = np.stack(images)

        # Add metadata
        zarr_container[dsname].attrs["resolution"] = (
            1,  # TODO: just a guess, need to replace with correct value later
            int(meta_data["YResolution"]),
            int(meta_data["XResolution"])
        )
