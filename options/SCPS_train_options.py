import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.common_options import CommonOptions

class SCPSTrainOptions(CommonOptions):
    def initialize(self, parser):        
        parser = CommonOptions.initialize(self, parser)        
        # model parameter for knowledge distillation 
        parser.add_argument('--train_dataroot', default='./SCPS_Dataset/trainConc/', help='path to image data folder')
        parser.add_argument('--train_listfile', type=str, default='trainConc.txt', help='text file of data file list')
        
        parser.add_argument('--inpainting', default=False, action='store_true', help='whether not to use inpainted images' )
        parser.add_argument('--inpaint_dataroot', default='./SCPS_Dataset/inpaintConc/', help='path to image data folder')        
        parser.add_argument('--inpaint_listfile', type=str, default='inpaintConc.txt', help='text file of data file list')                                           
                
        parser.add_argument('--val_dataroot', default='./SCPS_Dataset/testConc/', help='path to image data folder')
        parser.add_argument('--val_listfile', type=str, default='testConc.txt', help='text file of data file list')   
                
        parser.add_argument('--phase', type=str, default='train', help='[train | val | test]')        
        parser.add_argument('--batch_size', type=int, default=36, help='input data size, should be even number')        
        parser.add_argument('--flip', type=bool, default=False, help='whether image data flips or not')
        parser.add_argument('--crop', type=bool, default=True, help='whether image data crops or not. If true, assure crop size is needed')
        
        # train options
        # parser.add_argument('--n_epoch', type=int, default=1500, help='# of total epochs')
        # parser.add_argument('--save_freq', type=int, default=10, help='how often models are saved')                
        
        parser.add_argument('--n_epoch', type=int, default=100, help='# of total epochs')
        parser.add_argument('--save_freq', type=int, default=10, help='how often models are saved')                
                
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
        
        # parser.add_argument('--lp_epoch', type=int, default=1000, help='starting epoch to save the images in learning progress')
        # parser.add_argument('--lp_freq', type=int, default=10, help='freqeuncy how many times images during training')
        
        parser.add_argument('--lp_epoch', type=int, default=50, help='starting epoch to save the images in learning progress')
        parser.add_argument('--lp_freq', type=int, default=10, help='freqeuncy how many times images during training')
        
        parser.add_argument('--cps_weight', type=float, default=0.5, help='cps weight')                
        parser.add_argument("--with_tracking", action="store_true", help="Whether to load in all available experiment trackers from the environment and use them for logging.")        
        
        self.isTrain = True                                
        
        return parser

