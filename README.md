# EMASRN
This repository is Pytorch code for our proposed EMASRN.

## Test

#### Quick start
1. Download the testset
The testset can be downloaded from [[BaiduYun]](https://pan.baidu.com/s/18NsZHMbhSu14GxAw9jMgIw)(code:hl0v) and unzip it to ./results

2. cd to `EMASRN` and run **one of following commands** for evaluation:

   ```shell
   # EMASRN
   python test.py -opt options/test/test_example_x3.json
   python test.py -opt options/test/test_example_x4.json
   
3. Edit `./options/test/test_example_x3.json` or `./options/test/test_example_x4.json` for your needs
