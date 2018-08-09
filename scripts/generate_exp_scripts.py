import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, nargs='+', default=['CACD'])
parser.add_argument('--model', type=str, nargs='+', default=['hot', 'age', 'emb'])
parser.add_argument('--generator', '-G', type=str, nargs='+', default=['unet'])
parser.add_argument('--norm', type=str, nargs='+', default=['bn'])
parser.add_argument('--batch_size', type=int, nargs='+', default=[1])
parser.add_argument('--gpu_ids', type=int, default=0)
parser.add_argument('--date', type=str, default='0810')
parser.add_argument('--sh_name', type=str, default='run_batch.sh')

config = parser.parse_args()

# print(config)

base_args = {
    'CACD_hot': "--model faceaging --dataset_mode faceaging --age_binranges 11 21 31 41 51 --embedding_nc 5 --dataroot ./datasets/CACD --sourcefile_A ./sourcefiles/CACD_train.txt --name 0810_CACD_hot_unet_bn_bs1 --which_model_netD n_layers --n_layers_D 4 --which_model_netIP alexnet --pretrained_model_path_IP pretrained_models/alexnet.pth --which_model_netAC resnet18 --pretrained_model_path_AC ./pretrained_models/AC_resnet18_cacd5.pth --lambda_L1 0.0 --lambda_IP 1 --lambda_AC 1 --lambda_A 1 --pool_size 0 --loadSize 128 --fineSize 128 --fineSize_IP 224 --fineSize_AC 224 --display_port 8097 --display_freq 500 --print_freq 100 --niter 200 --niter_decay 0 --display_aging_visuals --max_dataset_size 10000",
    'CACD_age': "--model faceaging_age --dataset_mode faceaging_age --age_binranges 11 21 31 41 51 --embedding_nc 1 --dataroot ./datasets/CACD --sourcefile_A ./sourcefiles/train_pairs_m10_cacd.txt --name 0810_CACD_age_unet_bn_bs1 --which_model_netD n_layers --n_layers_D 4 --which_model_netIP alexnet --pretrained_model_path_IP pretrained_models/alexnet.pth --lambda_L1 0.0 --lambda_IP 1 --lambda_A 1 --pool_size 0 --loadSize 128 --fineSize 128 --fineSize_IP 224 --display_port 8097 --display_freq 500 --print_freq 100 --embedding_mean 5 --embedding_std 80 --niter 20 --niter_decay 0 --display_aging_visuals --max_dataset_size 100000",
    'CACD_emb': "--model faceaging_embedding --dataset_mode faceaging_embedding --age_binranges 11 21 31 41 51 --embedding_nc 1 --dataroot ./datasets/CACD --sourcefile_A ./sourcefiles/train_pairs_m10_cacd.txt --name 0810_CACD_emb_unet_bn_bs1 --which_model_netD n_layers --n_layers_D 4 --which_model_netIP alexnet --pretrained_model_path_IP pretrained_models/alexnet.pth --which_model_netE resnet18 --pooling_E avg --pretrained_model_path_E ./pretrained_models/E_resnet18_cacd.pth --lambda_L1 0.0 --lambda_IP 1 --pool_size 0 --loadSize 128 --fineSize 128 --fineSize_IP 224 --fineSize_E 224 --display_port 8097 --display_freq 500 --print_freq 100 --lambda_E 1 --embedding_mean -0.1311 --embedding_std 1.6006 --niter 20 --niter_decay 5 --display_aging_visuals --aging_visual_embedding_path ./fixed_features/fixed_features_cacd.npy --max_dataset_size 100000",
    'UTK_hot':  "--model faceaging --dataset_mode faceaging --age_binranges 1 21 41 61 81 --embedding_nc 5 --dataroot ./datasets/UTK --sourcefile_A ./sourcefiles/UTK_train.txt --name 0810_UTK_hot_unet_bn_bs1 --which_model_netD n_layers --n_layers_D 4 --which_model_netIP alexnet --pretrained_model_path_IP pretrained_models/alexnet.pth --which_model_netAC alexnet_lite --pretrained_model_path_AC ./pretrained_models/AC_alexnet_lite_avg_utk5.pth --pooling_AC avg --lambda_L1 0.0 --lambda_IP 1 --lambda_AC 1 --lambda_A 1 --pool_size 0 --loadSize 128 --fineSize 128 --fineSize_IP 224 --fineSize_AC 224 --display_port 8097 --display_freq 500 --print_freq 100 --niter 200 --niter_decay 0 --display_aging_visuals --max_dataset_size 10000",
    'UTK_age':  "--model faceaging_age --dataset_mode faceaging_age --age_binranges 1 21 41 61 81 --embedding_nc 1 --dataroot ./datasets/UTK --sourcefile_A ./sourcefiles/train_pairs_m10_utk.txt --name 0810_UTK_age_unet_bn_bs1 --which_model_netD n_layers --n_layers_D 4 --which_model_netIP alexnet --pretrained_model_path_IP pretrained_models/alexnet.pth --lambda_L1 0.0 --lambda_IP 1 --lambda_A 1 --pool_size 0 --loadSize 128 --fineSize 128 --fineSize_IP 224 --display_port 8097 --display_freq 500 --print_freq 100 --embedding_mean 0 --embedding_std 100 --niter 20 --niter_decay 0 --display_aging_visuals --max_dataset_size 100000",
    'UTK_emb':  "--model faceaging_embedding --dataset_mode faceaging_embedding --age_binranges 1 21 41 61 81 --embedding_nc 1 --dataroot ./datasets/UTK --sourcefile_A ./sourcefiles/train_pairs_m10_utk.txt --name 0810_UTK_emb_unet_bn_bs1 --which_model_netD n_layers --n_layers_D 4 --which_model_netIP alexnet --pretrained_model_path_IP pretrained_models/alexnet.pth --which_model_netE resnet18 --pooling_E avg --pretrained_model_path_E ./pretrained_models/E_resnet18_utk.pth --lambda_L1 0.0 --lambda_IP 1 --pool_size 0 --loadSize 128 --fineSize 128 --fineSize_IP 224 --fineSize_E 224 --display_port 8097 --display_freq 500 --print_freq 100 --lambda_E 1 --embedding_mean 10 --embedding_std 30 --niter 20 --niter_decay 5 --display_aging_visuals --aging_visual_embedding_path ./fixed_features/fixed_features_utk.npy --max_dataset_size 100000",
}
G_args = {
    'unet': 'unet_128',
    'resnet': 'resnet_6blocks'
}
norm_args = {
    'bn': 'batch',
    'in': 'instance'
}

with open(config.sh_name, 'w+') as f:
    f.write('set -ex\n')
    f.write('\n')
    for dataset in config.dataset:
        for model in config.model:
            arg_base = base_args['_'.join([dataset, model])]
            for G in config.generator:
                for norm in config.norm:
                    for bs in config.batch_size:
                        arg_G = '--which_model_netG ' + G_args[G]
                        arg_norm = '--norm_G ' + norm_args[norm]
                        arg_bs = '--batchSize ' + str(bs)
                        if bs > 1:
                            arg_extra = '--display_freq 100'
                        else:
                            arg_extra = ''
                        exp_name = '_'.join([config.date, dataset, model, G, norm, 'bs%d'%bs])
                        arg_name = '--name '+exp_name
                        args = ' '.join([arg_base, arg_G, arg_norm, arg_bs, arg_extra, arg_name])
                        command = 'CUDA_VISIBLE_DEVICES=%d '%config.gpu_ids+'python train.py '+args
                        f.write(command+'\n')
                        f.write('\n')
