import gunpowder as gp
import numpy as np
import torch
from gunpowder.torch import Train
from lsd.train.gp import AddLocalShapeDescriptor
from tqdm import tqdm
from livseg import MtlsdModel, WeightedMSELoss


def train(
    crops,
    voxel_size=(198, 45, 45),
    shape=(32, 64, 64),
    output_shape=(12, 24, 24),
    batch_size=8
):
    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate(shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    # Define the different types of data
    raw = gp.ArrayKey('RAW')
    label = gp.ArrayKey('LABEL')
    gt_lsds = gp.ArrayKey('GT_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    # Create a source for each crop
    sources = tuple(
        gp.ZarrSource(
            crop,
        {
        raw: 'raw',
        label: 'label'
        },
        {
        raw: gp.ArraySpec(interpolatable=True),
        label: gp.ArraySpec(interpolatable=False)
        }) + gp.RandomLocation(min_masked=0.01, mask=label)
        for crop in crops
    )
    normalize = gp.Normalize(raw)

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

    pipeline = sources + gp.RandomProvider()
    pipeline += normalize
    pipeline += simple_augment
    pipeline += intensity_augment
    pipeline += gp.GrowBoundary(label)
    pipeline += AddLocalShapeDescriptor(
        label,
        gt_lsds,
        lsds_mask=lsds_weights,
        sigma=120.0  # TODO Make configurable?
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

    # Add a channel dimension?
    pipeline += gp.Unsqueeze([raw])

    # stack
    pipeline += gp.Stack(batch_size)

    # Create the request
    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(label, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    # Defining variables for the model
    # TODO Make configurable? 
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

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # adding the log
    log_dir = './tensorboard_summaries'  # TODO: make configurable
    log_every = 100  # TODO: make configurable

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
        },
        log_dir=log_dir,
        log_every=log_every
    )

    pipeline += train

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)


    # creating the snapshot
    dataset_names = {
        raw: 'Raw',
        pred_lsds: 'Predicted_Lsds',
        gt_lsds: 'Ground_Truth_Lsds',
        lsds_weights: 'Lsds_Weights',
        pred_affs: 'Predicted_Affinities',
        gt_affs: 'Ground_Truth_Affinities',
        affs_weights: 'Affinities_Weights'
    }
    output_dir = './snapshots_folder'  # TODO: make configurable
    output_filename = 'Snapshot.zarr'  # TODO: make configurable
    every = 100  # TODO: make configurable
    pipeline += gp.Snapshot(
        dataset_names,
        output_dir=output_dir,
        output_filename=output_filename,
        every=every,
        additional_request=None,
        compression_type=None,
        dataset_dtypes=None,
        store_value_range=False
    )

    # TODO: fix the number of iterations
    iterations = 100000

    with gp.build(pipeline):
        progress = tqdm(range(iterations))
        for i in progress:
            batch = pipeline.request_batch(request)

        #     start = request[label].roi.get_begin()/voxel_size
        #     end = request[label].roi.get_end()/voxel_size
        # progress.set_description(f'Training iteration {i}')


if __name__ == "__main__":
    # TODO Load crops from a configuration file
    crops = [
        "/groups/funke/funkelab/livseg/data/training_data/cc1.zarr",
        "/groups/funke/funkelab/livseg/data/training_data/moa1.zarr"
    ]

    train(crops)