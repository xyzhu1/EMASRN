# EMASRN
This repository is Pytorch code for our proposed EMASRN.

## Test

#### Quick start
The testset can be downloaded from  [[GoogleDrive]](https://drive.google.com/file/d/1Dsb_-OH0CeSJVjvP9A4bh2_IBQh9R-ja/view) or [[BaiduYun]](https://pan.baidu.com/s/1fIGBulcWll8MzaS87D_kPQ)(code:6qta) and unzip it to ./result

cd to `EMASRN` and run **one of following commands** for evaluation on *Set5*:

   ```shell
   # SRFBN
   python test.py -opt options/test/test_example_x3.json
   python test.py -opt options/test/test_example_x4.json
