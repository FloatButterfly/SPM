export HDF5_USE_FILE_LOCKING='FALSE'
# multi gpu
python -m torch.distributed.launch --nproc_per_node=4  train_spm_codec.py --name name --train_dataroot train_dataroot --test_dataroot val_dataroot --style_nc 64 --batchSize n --niter x --niter_decay y --checkpoints_dir ckpt_root --lr 3e-4 --no_TTUR --label_nc 18 --no_instance --contain_dontcare_label --dataset_mode celeba --continue_train --which_epoch num --init_type "kaiming"  --total_step 100000 --with_test --k_latent 0 --k_L1 0 --k_feat 10 --k_gan 1 --k_vgg 10 --GANmode

