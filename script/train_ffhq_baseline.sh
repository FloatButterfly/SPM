export HDF5_USE_FILE_LOCKING='FALSE'
# multi gpu
python -m torch.distributed.launch --nproc_per_node=4  train_spm_codec.py --name name --train_dataroot train_dataroot --test_dataroot val_dataroot --style_nc 64 --batchSize n --niter x --niter_decay y --checkpoints_dir ckpt_root --lr 3e-4 --no_TTUR --label_nc 18 --no_instance --contain_dontcare_label --dataset_mode celeba --continue_train --which_epoch num --init_type "kaiming"  --total_step 100000 --with_test --k_latent 0 --k_L1 0 --k_feat 10 --k_gan 1 --k_vgg 10 --GANmode

#python train_spm_codec.py --name ffhq_baseline --train_dataroot F://Dataset/CelebAMask-HQ/train/ --test_dataroot F://Dataset/CelebAMask-HQ/val/ --style_nc 64 --batchSize 1 --niter 3 --niter_decay 3 --checkpoints_dir ../NIPS_checkpoint/sean/ --lr 4e-4 --no_TTUR --label_nc 18 --no_instance --contain_dontcare_label --dataset_mode celeba --continue_train --which_epoch 10 --init_type "kaiming"  --total_step 100000 --k_latent 0 --k_L1 0 --k_feat 10 --k_gan 1 --k_vgg 10 --GANmode
#
# --norm_G spectralstylesyncbatch3x3
# --use_id
# /userhome/anaconda3/envs/pytorch/bin/python train_sean_codec.py --name celeba_64 --train_dataroot ../datasets/CelebAMask-HQ/train/ --test_dataroot ../datasets/CelebAMask-HQ/val/ --style_nc 64 --batchSize 3 --niter 1 --niter_decay 0 --checkpoints_dir ../checkpoint/sean_codec/ --lr 1e-5 --no_TTUR --label_nc 18 --no_instance --with_entropy --gpu_ids '0,1,2' --contain_dontcare_label --continue_train --which_epoch latest --k_mse 0 --k_lpips 0 --lmbda 100 --GANmode --train_entropy --ablation
# standard
# /userhome/anaconda3/envs/pytorch/bin/python train_sean_codec.py --name celeba_64 --train_dataroot ../datasets/CelebAMask-HQ/train/ --test_dataroot ../datasets/CelebAMask-HQ/val/ --style_nc 64 --batchSize 6 --niter 30 --niter_decay 20 --checkpoints_dir ../checkpoint/sean_codec/ --lr 1e-4 --no_TTUR --continue_train --label_nc 18 --no_instance --with_entropy --which_epoch latest 
