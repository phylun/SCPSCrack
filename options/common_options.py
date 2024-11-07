'''
This code of 'options' is inspired by Jun-Yan Zhu*, Taesung Park*, Phillip Isola, and Alexei A. Efros. 
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", 
in IEEE International Conference on Computer Vision (ICCV), 2017.

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options

'''

import argparse
import os,sys
import torch
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import util

class CommonOptions():

    def __init__(self) :
        self.initialized = False        
        seednum=2024        
        torch.manual_seed(seednum)
        random.seed(seednum)
        np.random.seed(seednum)
            
        # GPU SEED fixed
        torch.cuda.manual_seed(seednum)
        torch.cuda.manual_seed_all(seednum)
        
        # CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def initialize(self, parser):

        # common parameter        
        parser.add_argument('--name', type=str, default='GenConc100', help='name of experiment [concrete | asphalt]')        
        parser.add_argument('--save_dir', type=str, default='./outputs', help='where moodels are saved')        
        parser.add_argument('--results_dir', type=str, default='./results', help='where result images are saved')
        parser.add_argument('--prog_dir', type=str, default='./learning_progress', help='where results in the middle of learning are saved')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # model parameter for supervised learning        
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channel: 3 for RGB and 1 for binary')
        parser.add_argument('--output_nc', type=int, default=2, help='# of output segmentation channel: 2')                                
        parser.add_argument('--Snet', type=str, default='PoolformerS36', help='specify segmentation network [FRRNA | CGNet | LEDNet | FDDWNet | ERFNet | DDRNet | RegSeg | PIDNet | Deeplabv3p]')                
        
                
        # dataset parameter 
        parser.add_argument('--num_threads', type=int, default=4, help='# of threads for data load')        
        parser.add_argument('--image_size', type=int, default=448, help='input image size')
        parser.add_argument('--label_size', type=int, default=448, help='lable image size')
        parser.add_argument('--max_dataset_size', type=float, default=float('inf'), help='Maximum number of samples in dataset directory')        
        parser.add_argument('--crop_size', type=int, default=224, help='croping size')
        
        # accelerate parameter
        parser.add_argument("--bf16", action="store_true", help="If passed, will use FP16 training.")        
        parser.add_argument("--mixed_precision", type=str, default='bf16', choices=["no", "fp16", "bf16", "fp8"],
                            help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
        parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")


        self.initialized = True
        return parser

    def setting_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.save_dir, opt.name)
        util.mkdirs(expr_dir)
        prog_dir = os.path.join(opt.prog_dir, opt.name)
        util.mkdirs(prog_dir)
        results_dir = os.path.join(opt.results_dir, opt.name)
        util.mkdirs(results_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.setting_options()
        self.isTrain = True if opt.phase == 'train' else False
            
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)
        

        self.opt = opt
        return self.opt


if __name__ == '__main__':
    print('Common Options')
    ComOpt = CommonOptions()    
    Opt = ComOpt.parse()

