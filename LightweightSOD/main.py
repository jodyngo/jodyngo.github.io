import argparse
import os
from datasets.dataset import get_loader
from solver import Solver
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()
if cuda:
    torch.backends.cudnn.benchmark = True
    current_device = torch.cuda.current_device()
    print("Running on", torch.cuda.get_device_name(current_device))
else:
    print("Running on CPU")


def get_test_info(sal_mode='e'):
    if sal_mode == 'e':
        image_root = './data/ECSSD/Imgs/'
        image_source = './data/ECSSD/test.lst'
    elif sal_mode == 'p':
        image_root = './data/PASCALS/Imgs/'
        image_source = './data/PASCALS/test.lst'
    elif sal_mode == 'd':
        image_root = './data/DUTOMRON/Imgs/'
        image_source = './data/DUTOMRON/test.lst'
    elif sal_mode == 'h':
        image_root = './data/HKU-IS/Imgs/'
        image_source = './data/HKU-IS/test.lst'
    elif sal_mode == 's':
        image_root = './data/SOD/Imgs/'
        image_source = './data/SOD/test.lst'
    elif sal_mode == 't':
        image_root = './data/DUTS-TE/Imgs/'
        image_source = './data/DUTS-TE/test.lst'
    elif sal_mode == 'm_r':  # for speed test
        image_root = './data/MSRA/Imgs_resized/'
        image_source = './data/MSRA/test_resized.lst'

    return image_root, image_source


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        config.test_root, config.test_list = get_test_info(config.sal_mode)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    # resnet_path = 'datasets/pretrained/resnet18-5c106cde.pth'
    resnet_path = 'datasets/pretrained/resnet50_caffe.pth'
    # resnet_path = './networks/p2t_small.pth'
    # resnet_path = 'datasets/pretrained/vgg16_20M.pth'
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)  # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=5e-4)  # Weight decay 0.01 mob
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    # Training settings
    parser.add_argument('--arch', type=str, default='resmot')  # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--iter_size', type=int, default=20)
    parser.add_argument('--show_every', type=int, default=200)

    # Train data
    parser.add_argument('--train_root', type=str, default='/home/thungo/PSAMNet/data/DUTS/DUTS-TR')
    parser.add_argument('--train_list', type=str, default='/home/thungo/PSAMNet/data/DUTS/DUTS-TR/train_pair.lst')

    # Testing settings
    parser.add_argument('--model', type=str, default=None)  # Snapshot
    parser.add_argument('--test_fold', type=str, default=None)  # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='e')  # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # Get test set info
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list

    main(config)
