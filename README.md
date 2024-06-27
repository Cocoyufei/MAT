

## Dataset preparation
- Download S3DIS [dataset](https://drive.google.com/uc?export=download&id=1KUxWagmEWnvMhEb4FRwq2Mj0aa3U3xUf) and symlink the paths to them as follows:

     ```
     mkdir -p dataset
     ln -s /path_to_s3dis_dataset dataset/s3dis
     ```

## Usage
- Shape classification on ModelNet40
  - For now, please use paconv-codebase branch.
- Part segmentation on ShapeNetPart
  - For now, please use paconv-codebase branch.
- Semantic segmantation on S3DIS Area 5
  - Train

    - Specify the gpu used in config and then do training:

      ```
      sh tool/train.sh s3dis pointtransformer_repro
      ```

  - Test

    - Afer training, you can test the checkpoint as follows:

      ```
      CUDA_VISIBLE_DEVICES=0 sh tool/test.sh s3dis pointtransformer_repro
      ```
  ---


## Acknowledgement
The code is from the first author of [Point Transformer](https://arxiv.org/abs/2012.09164).

