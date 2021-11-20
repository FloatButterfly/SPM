# from path import cur_dir

import imageio
from torch.utils.data import DataLoader
import numpy as np
from lpips_pytorch import lpips
from data import *
from models import *
from options.test_options import TestOptions
from util import *
from torch import nn

opt = TestOptions().parse()

test_dataset = TrainPairedData(opt.dataroot, opt)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)
model = SPMModel(opt)
model.eval()
bpp = 0
avg_lpips = 0
avg_fid = 0
# test
n = min(len(test_loader), opt.how_many)
up = nn.Upsample(scale_factor=2, mode='bicubic')
lpips_list = []
with torch.no_grad():
    for i, data_i in enumerate(test_loader):
        if i * opt.batchSize >= opt.how_many:
            break

        generated, tex_bpp, tex_code = model(data_i, mode='val')

        lpips_score = lpips(opt, data_i['image'].cuda(), generated.cuda())
        lpips_list.append(lpips_score.item())

        bpp = bpp + tex_bpp
        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            input = ((data_i['image'][0] + 1) / 2.0 * 255.0).round_().clamp(0, 255)
            input = input.detach().cpu().float().numpy()
            input = np.transpose(input, (1, 2, 0))
            input = input.astype(np.uint8)

            generated = ((generated[0] + 1) / 2.0 * 255.0).round_().clamp(0, 255)
            generated = generated.detach().cpu().float().numpy()
            generated = np.squeeze(generated)
            generated = np.transpose(generated, [1, 2, 0])
            generated = generated.astype(np.uint8)
            path = os.path.join(opt.results_dir, opt.name)
            if not os.path.exists(path):
                os.makedirs(path)
            if opt.dataset_mode == 'ade20k':
                save_path = os.path.join(path, "decoded_%.4f_bpp_lpips_%.4f_" % (tex_bpp, lpips_score) +
                                         img_path[b].split('\\')[-1])
                input_savepath = os.path.join(path, "input_" + img_path[b].split('\\')[-1])
            elif opt.dataset_mode == 'celeba':
                save_path = os.path.join(path, "decoded_%.4f_bpp_lpips_%.4f_" % (tex_bpp, lpips_score) +
                                         img_path[b].split('\\')[-1])
                input_savepath = os.path.join(path, "input_" + img_path[b].split('\\')[-1])
            else:
                save_path = os.path.join(path, "decoded_%.4f_bpp_lpips_%.4f_" % (tex_bpp, lpips_score) +
                                         img_path[b].split('\\')[-1])
                input_savepath = os.path.join(path, "input_" + img_path[b].split('\\')[-1])
            imageio.imsave(save_path, generated)
            imageio.imsave(input_savepath, input)

            print("the bpp of texture code of image %s is : %.5f" % (save_path, tex_bpp))
            print("the lpips score is %.4f" % lpips_score)

            with open(os.path.join(path, "log.txt"), 'a') as f:
                f.write(
                    "image {}:\t bpp:{:>.4f}\t lpips:{:>.4f}\t \n".format(img_path,
                                                                          tex_bpp.item(),
                                                                          lpips_score.item()
                                                                          ))
    _lpips = np.asarray(lpips_list)
    avg_lpips = _lpips.mean()
    var_lpips = _lpips.var()
    print("average texture bpp is %.5f" % (bpp / n))
    print("average lpips score is %.5f" % (avg_lpips))
    print("variance of lpips score is %.5f" % (var_lpips))
    with open(os.path.join(path, "log.txt"), 'a') as f:
        f.write(
            "average bpp: {}\t average lpips:{}\t lpips variance:{} \t".format(bpp / n, avg_lpips, var_lpips) + '\n')
