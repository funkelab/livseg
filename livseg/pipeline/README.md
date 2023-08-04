FULL DATASET OBTAINED FROM FELICIANO LAB

1. Run convert_to_zarr.py with the needed arguments 
    - First argument should be the source directory to where the dapi data is located
    - Second argument should be the target directory to where the converted files should go
    - Third argument should be the source directory to where all the other channels are located
2. Generate a crop using get_crop.py
    - The size and the center of the crop need to be defined
3. Send it to the professional annotaters
    - if given through AMIRA: it will generate tiffs
4. Convert back to zarrs using FIJI (ask Caleb if you're unsure)
    - name this file as 'label' and put under zarr container that also contains the raw data of that crop
5. Train the model with the train_template.py by adding the new crop
    - make sure to also add a new zarrsource 
6. Run the predict_template.py on a new crop and make sure to change the directories to match up with the new crop/experiment
    - target_dir + "/" + output_file contains the Raw, Pred_LSDS, and Pred_AFFS
    - target_dir2 contains the final 3d segmentations thats outputted from the post processingx