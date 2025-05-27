from models_x import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import os
from torchvision import transforms
import glob

def get_latest_ckpt(path_pattern):
    ckpts = glob.glob(path_pattern)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found for pattern: {path_pattern}")
    return max(ckpts, key=os.path.getctime)  # or sort by filename if preferred

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='data/test')
    parser.add_argument('--dataset', type=str, default='Flickr1024_patches_masked_irregular_test')
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

def test(test_loader, cfg):
    net = PASSRnet(cfg.scale_factor).to(cfg.device)
    cudnn.benchmark = True
    # pretrained_dict = torch.load('./log_x' + '/PASSRnet_x' + str(cfg.scale_factor) + '.pth')
    ckpt_path = get_latest_ckpt(f'./log_x/PASSRnet_epoch*.pth.tar')
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda:0'))
    net.load_state_dict(checkpoint['state_dict'])

    psnr_list_L = []
    psnr_list_R = []

    with torch.no_grad():
        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(test_loader):
            HR_left, HR_right, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
            scene_name = test_loader.dataset.file_list[idx_iter]

            (SR_left, SR_right) = net(LR_left, LR_right, is_training=0)
            SR_left = torch.clamp(SR_left, 0, 1)
            SR_right = torch.clamp(SR_right, 0, 1)

            psnr_list_L.append(cal_psnr(HR_left[:,:,:,64:], SR_left[:,:,:,64:]))
            psnr_list_R.append(cal_psnr(HR_right[:,:,:,64:], SR_right[:,:,:,64:]))


            ## save results
            output_dir = f'results_x/{cfg.dataset}/{scene_name}'
            os.makedirs(output_dir, exist_ok=True)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save(f'{output_dir}/recon_L.png')
            SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
            SR_right_img.save(f'{output_dir}/recon_R.png')

        ## print results
        print(cfg.dataset + ' mean psnr_L: ', float(np.array(psnr_list_L).mean()))
        print(cfg.dataset + ' mean psnr_R: ', float(np.array(psnr_list_R).mean()))


def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    result = test(test_loader, cfg)
    return result

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
