
## Procedure
1. valid_dataset.zip
2. Preprocessing 
    1. `multiproc_resample.py` to convert sample rate of audio files to 24000 Hz
       ```
       python multiproc_resample.py --src_path <path_to_train> --dst_path <path_to_train_24000>
       python multiproc_resample.py --src_path <path_to_validation> --dst_path <path_to_val_24000>
       python multiproc_resample.py --src_path <path_to_test> --dst_path <path_to_test_24000>
       ```
    2. `chunkify.py` to make chunks from train,validation and test files
        ```
        python chunkify.py --src_dir <path_to_train_24000> --tgt_dir <path_to_train_24000_chunks>
        python chunkify.py --src_dir <path_to_val_24000> --tgt_dir <path_to_val_2400_chunks>
        python chunkify.py --src_dir <path_to_test_24000> --tgt_dir <path_to_test_2400_chunks>
        ```
    3. `make_chunks_manifests.py` to make manifest files used for training.
        ```
        python make_chunks_manifests.py --dataset_path <dataset/valid_dataset/> --train_csv <metadata/train.csv> --val_csv <metadata/validation.csv> --test_csv <metadata/test.csv> --train_chunks_dir <train_24000_chunks> --val_chunks_dir <validation_24000_chunks> --test_chunks_dir <test_24000_chunks> --output_dir <chunks_meta_dir>
        ```
       This will store absolute file paths in the generated train_chunk, val_chunk and test_chunk csv files.

3. Training and validation

4. Evaluation on Evaluation set
   


## Acknowledgements


## References

