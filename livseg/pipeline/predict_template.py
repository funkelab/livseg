import gunpowder as gp
import zarr
import numpy as np
from livseg import MtlsdModel
import waterz

from scipy.ndimage import label
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed

raw_data = "/groups/funke/funkelab/livseg/data/crops.zarr/bigger_portal_vein_crop.zarr"
raw_file = 'raw'
checkpoint = "/groups/funke/funkelab/livseg/experiments/20230801_more_training/model_checkpoint_100000"

# directory for predictions -- Raw Pred_LSDS Pred_AFFS
target_dir = "/groups/funke/funkelab/livseg/experiments/20230801_more_training/predictions"
output_file = "portal_crop.zarr"

# directory for 3d segmentations post processing
target_dir2 = "/groups/funke/funkelab/livseg/experiments/20230801_more_training/fragments.zarr"

zarrfile = zarr.open(raw_data + "/raw", "r")
offset = zarrfile.attrs["offset"]

voxel_size = gp.Coordinate((198, 45, 45))
size = gp.Coordinate((32, 64, 64))
output_shape = gp.Coordinate((12, 24, 24))

input_size = size*voxel_size
output_size = output_shape*voxel_size

def predict(checkpoint, raw_data, raw_file):
    raw = gp.ArrayKey("RAW")
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(pred_affs, output_size)

    # scan_request[raw] = gp.Roi(offset, voxel_size * size)

    context = (input_size - output_size) / 2

    raw_source = gp.ZarrSource(
        raw_data,
        {raw: raw_file},
        {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(raw_source):
        total_input_roi = raw_source.spec[raw].roi
        total_output_roi = raw_source.spec[raw].roi.grow(-context, -context)

    lsd_shape = total_output_roi.get_shape() / voxel_size
    aff_shape = total_output_roi.get_shape() / voxel_size

    # generating the zarr file for saving
    zarrfile = zarr.open(target_dir + "/" + output_file, 'w')

    zarrfile.create_dataset('Raw', shape= total_input_roi.get_shape() / voxel_size)
    zarrfile.create_dataset('Pred_LSDS', shape = (10, lsd_shape[0], lsd_shape[1], lsd_shape[2]))
    zarrfile.create_dataset('Pred_AFFS', shape = (3, aff_shape[0], aff_shape[1], aff_shape[2]))

    zarrfile['Raw'].attrs['resolution'] = voxel_size
    zarrfile['Pred_LSDS'].attrs['resolution'] = voxel_size
    zarrfile['Pred_AFFS'].attrs['resolution'] = voxel_size

    zarrfile['Raw'].attrs['offset'] = total_input_roi.offset
    zarrfile['Pred_LSDS'].attrs['offset'] = total_output_roi.offset
    zarrfile['Pred_AFFS'].attrs['offset'] = total_output_roi.offset

    in_channels = 1
    num_fmaps = 16
    fmap_inc_factor = 2
    downsample_factors = [(1, 2, 2), (1, 2, 2)]
    num_levels = len(downsample_factors) + 1
    kernel_size_down = [[(3, 3, 3), (3, 3, 3)]] * num_levels
    kernel_size_up = [[(3, 3, 3), (3, 3, 3)]] * (num_levels - 1)
    constant_upsample = True

    lr = 1e-5

    model = MtlsdModel(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        constant_upsample=constant_upsample
    )

    model.eval()

    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_lsds,
            1: pred_affs})

    pipeline = raw_source
    pipeline += gp.Normalize(raw)

    pipeline += gp.Unsqueeze([raw])

    pipeline += gp.Stack(1)

    pipeline += predict

    pipeline += gp.Squeeze([raw])

    pipeline += gp.Squeeze([raw, pred_lsds, pred_affs])

    dataset_names = {
        raw: 'Raw',
        pred_lsds: 'Pred_LSDS',
        pred_affs: 'Pred_AFFS',
    }

    pipeline += gp.ZarrWrite(
        dataset_names = dataset_names,
        output_dir = target_dir,
        output_filename = output_file
    )

    pipeline += gp.Scan(scan_request)


    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred_lsds] = total_output_roi
    predict_request[pred_affs] = total_output_roi

    print(offset)
    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    # print(
    #     f"\tRaw: {batch[raw].data}, \tPred LSDS: {batch[pred_lsds].data}, \tPred Affs: {batch[pred_affs].data}")
    
    return batch[raw].data, batch[pred_lsds].data, batch[pred_affs].data

raw, pred_lsds, pred_affs = predict(checkpoint, raw_data, raw_file)

def watershed_from_boundary_distance(
        
        boundary_distances,
        boundary_mask,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)

    ret = (fragments.astype(np.uint64), n + id_offset)

    return ret

def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        id_offset=0,
        min_seed_distance=3):

    mean_affs = 0.5*(affs[1] + affs[2])

    fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
  
    boundary_mask = mean_affs>0.5*max_affinity_value
    boundary_distances = distance_transform_edt(boundary_mask)

    ret = watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=id_offset,
        min_seed_distance=min_seed_distance)

    return ret

def get_segmentation(affinities, threshold):

    fragments = watershed_from_affinities(affinities)[0]
    thresholds = [threshold]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    segmentation = next(generator)

    return segmentation

# TODO: use all 3 dimensions (change line 219)
ws_affs = np.stack([
    np.zeros_like(pred_affs[0]),
    pred_affs[0],
    pred_affs[1]]
)

threshold = 0.9

segmentation = get_segmentation(ws_affs, threshold)

zarr_file = zarr.open(target_dir2, 'w')

zarr_file['segmentation'] = segmentation

# TODO: make voxel size a tuple
zarr_file['segmentation'].attrs['resolution'] = (198, 45, 45)

zarr_file['segmentation'].attrs['offset'] = offset