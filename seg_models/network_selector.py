from seg_models.CGNet import CGNet
from seg_models.FDDWNet import FddwNet
from seg_models.ERFNet import ERFNet
from seg_models.DDRNet_39 import DualResNet_imagenet
from seg_models.RegSeg import RegSeg
from seg_models.convnext import Convnext
from seg_models.poolformer import Poolformer
from seg_models.swinformer import Swinformer

'''
# This Convnext, poolformer models were implemented by https://github.com/sithu31296/semantic-segmentation
'''
def network_selection(opt):
            
    
        
    if opt.Snet == 'CGNet':
        model = CGNet(n_classes=opt.output_nc)    
    elif opt.Snet == 'FDDWNet':
        model = FddwNet(classes=opt.output_nc, in_ch=opt.input_nc)
    elif opt.Snet == 'ERFNet':
        model = ERFNet(num_classes=opt.output_nc)
    elif opt.Snet == 'DDRNet':
        model = DualResNet_imagenet()
    elif opt.Snet == 'RegSeg':
        model = RegSeg("exp48_decoder26", opt.output_nc)        
    elif opt.Snet == 'ConvnextT':
        model = Convnext('T', num_classes=opt.output_nc)
    elif opt.Snet == 'ConvnextS':
        model = Convnext('S', num_classes=opt.output_nc)
    elif opt.Snet == 'PoolformerS24':
        model = Poolformer('S24', num_classes=opt.output_nc)
    elif opt.Snet == 'PoolformerS36':
        model = Poolformer('S36', num_classes=opt.output_nc)        
    elif opt.Snet == 'SwinformerT':
        model = Swinformer('T', num_classes=opt.output_nc)    
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % opt.Snet)
    
    return model
    
    