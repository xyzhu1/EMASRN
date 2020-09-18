import argparse, random
from tqdm import tqdm
#
import torch

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# from torchstat import stat  #717
# # # # # # from networks.awsrn_arch import AWSRN
# # # # # # # # from networks.macn_arch import MACN
# from networks.imdn_arch import IMDN
# from networks.cfsrcnn_arch import CFSRCNN
# # # # # #
# from networks.msrn_arch import MSRN
# from networks.carn_arch import CARN
# from networks.zxynet_arch import ZXYNET
# # # from networks.mvpnet_arch import MVPNET
# # # # # # # #
# net = ZXYNET(in_channels=3, out_channels=3,
#            num_features=32, num_steps=3, num_groups=3,
#             upscale_factor=4)
# stat(MSRN(), (3, 320, 180))
# 
# 
# from thop import profile
# input = torch.randn(1, 3, 320, 180)
# macs, params = profile(MSRN(), inputs=(input, ))
# print('Total macc:{}, Total params: {}'.format(macs, params))



def main():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)

    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)
    print("===> Random Seed: [%d]"%seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader制造训练集和验证集
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('===> Train Dataset: %s   Number of images: [%d]' % (train_set.name(), len(train_set)))
            if train_loader is None: raise ValueError("[Error] The training data does not exist")

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val_set.name(), len(val_set)))

        else:
            raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)


    solver = create_solver(opt)#创建一个对象

    scale = opt['scale']#放大的倍数
    model_name = opt['networks']['which_model'].upper()#模型的名字

    print('===> Start Train')
    print("==================================================")


    solver_log = solver.get_current_log()#这是一个字典


    NUM_EPOCH = int(opt['solver']['num_epochs'])#运行的代数字
    start_epoch = solver_log['epoch']#从哪个epoch开始

    print("Method: %s || Scale: %d || Epoch Range: (%d ~ %d)"%(model_name, scale, start_epoch, NUM_EPOCH))

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print('\n===> Training Epoch: [%d/%d]...  Learning Rate: %f'%(epoch,
                                                                      NUM_EPOCH,
                                                                      solver.get_current_learning_rate()))

        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        t0 = time.time()
        train_loss_list = []


        with tqdm(total=len(train_loader), desc='Epoch: [%d/%d]'%(epoch, NUM_EPOCH), miniters=1) as t:#进度条
            for iter, batch in enumerate(train_loader):
                solver.feed_data(batch)#送入数据
                iter_loss = solver.train_step()#计算出损失
                batch_size = batch['LR'].size(0)
                train_loss_list.append(iter_loss*batch_size)#计算出每个batch的损失
                t.set_postfix_str("Batch Loss: %.4f" % iter_loss)
                t.update()

        solver_log['records']['train_loss'].append(sum(train_loss_list)/len(train_set))
        solver_log['records']['lr'].append(solver.get_current_learning_rate())

        t1 = time.time()

        print('\nEpoch: [%d/%d]   Avg Train Loss: %.6f    Everyone Epoch Timer: %.4f sec'  % (epoch,
                                                    NUM_EPOCH,
                                                    sum(train_loss_list)/len(train_set),(t1 - t0)))#计算这个epoch的平均损失

        print('===> Validating...',)

        #######
        # solver.save_checkpoint(epoch, True)  # 保存最新的和最好的pth文件

        psnr_list = []
        ssim_list = []
        val_loss_list = []

        for iter, batch in enumerate(val_loader):
            solver.feed_data(batch)
            iter_loss = solver.test()
            val_loss_list.append(iter_loss)

            # calculate evaluation metrics
            visuals = solver.get_current_visual()#返回LR和SR图像
            psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)#计算PSNR和SSIM参数
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if opt["save_image"]:#是否存储对应的图像数据
                solver.save_current_visual(epoch, iter)

        solver_log['records']['val_loss'].append(sum(val_loss_list)/len(val_loss_list))
        solver_log['records']['psnr'].append(sum(psnr_list)/len(psnr_list))
        solver_log['records']['ssim'].append(sum(ssim_list)/len(ssim_list))

        # record the best epoch
        epoch_is_best = False
        if solver_log['best_pred'] < (sum(psnr_list)/len(psnr_list)):
            solver_log['best_pred'] = (sum(psnr_list)/len(psnr_list))
            epoch_is_best = True
            solver_log['best_epoch'] = epoch

        print("[%s] PSNR: %.2f   SSIM: %.4f   Loss: %.6f   Best PSNR: %.2f in Epoch: [%d]" % (val_set.name(),
                                                                                              sum(psnr_list)/len(psnr_list),
                                                                                              sum(ssim_list)/len(ssim_list),
                                                                                              sum(val_loss_list)/len(val_loss_list),
                                                                                              solver_log['best_pred'],
                                                                                              solver_log['best_epoch']))

        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best)#保存最新的和最好的pth文件
        solver.save_current_log()#保存当前的训练数据记录

        # update lr
        solver.update_learning_rate(epoch)#改变学习率

    print('===> Finished !')


if __name__ == '__main__':
    main()
