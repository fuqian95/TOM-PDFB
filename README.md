# TOM-PDFB

## Requirements
The codes are tested in the following environment:
- python 3.8
- pytorch 1.10.1
- CUDA 10.2 & CuDNN 8

~~~python
pip3 install -r requirements.txt
~~~

## Performance

The model is trained on Composition-1K train dataset.
| Models | SAD | MSE(x10^(-3)) | Grad | Conn | Link|
|  ----  | ----  |  ----  | ----  |  ----  | ----  |
| TOM-PDFB | 21.46 | 3.36 | 7.38 | 16.09 | [Google Drive](https://drive.google.com/file/d/13vXPZbK8bePZaBtPRIXCVRKYmFN4l_ty/view?usp=sharing) |


## Testing on Composition-1K
Download the model file 'checkpoints/' and place it in the root directory.
Update the file paths for input data, output data, pre-trained model, etc., in the 'Composition1k.toml' configuration file.

1.Run the test code
~~~python
python3 inference.py
~~~

2.Evaluate the results by the official evaluation MATLAB code ./DIM_evaluation_code/evaluate.m (provided by [Deep Image Matting](https://sites.google.com/view/deepimagematting))

## Testing on Transparent-460

Obtain the dataset from [TransMatting](https://github.com/AceCHQ/TransMatting). 
Run the test code. 
Evaluate the results using the evaluation code provided in Transparent-460.

## Acknowledgment
This repo borrows code from several repos, like [GCA](https://github.com/Yaoyi-Li/GCA-Matting) and [MatteFormer](https://github.com/webtoon/matteformer.git)
