## Requirements
- Python 3.6 (Anaconda is recommended)
- skimage
- imageio
- Pytorch (Pytorch version >=1.2.0 is recommended)
- tqdm 
- pandas
- cv2 (pip install opencv-python)

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

## Citation
Please kindly cite our paper when using this project for your research.
```
@ARTICLE{9427111,
  author={Zhu, Xiangyuan and Guo, Kehua and Ren, Sheng and Hu, Bin and Hu, Min and Fang, Hui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Lightweight Image Super-Resolution with Expectation-Maximization Attention Mechanism}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2021.3078436}}
```
