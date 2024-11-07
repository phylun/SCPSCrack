from PIL import Image
import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from utils.eval_pre_rec_f1 import compute_mean_iou, f1score

from utils.util import GetPalletePIL, get_DenormTensor
from seg_models.network_selector import network_selection

from options.test_options import TestOptions

from accelerate import Accelerator, DistributedDataParallelKwargs

from utils.data_load_utils import get_NormTensor, PathFileDataset, func_txtpathfile_read
import torchvision.transforms as transforms


def test_function(opt):

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # no with_tracking
    accelerator = Accelerator(cpu=opt.cpu, mixed_precision=opt.mixed_precision, kwargs_handlers=[ddp_kwargs])

    result_dir = os.path.join(opt.results_dir, opt.name)    
    if not isinstance(opt.image_size, (list, tuple)):
        image_size = (opt.image_size, opt.image_size)    

    test_file_names = func_txtpathfile_read(opt.test_dataroot, opt.test_listfile)
    image_tf, label_tf = get_NormTensor()
    test_dataset = PathFileDataset(test_file_names, crop_size=image_size, image_transform=image_tf, label_trainsform=label_tf, phase='test')    
    test_dataloader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)
    
    denorm_transform = get_DenormTensor(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])    

    model_l = network_selection(opt)    
    model_r = network_selection(opt)        

    test_dataloader, model_l, model_r = accelerator.prepare(test_dataloader, model_l, model_r)

    model_folder = os.path.join(opt.save_dir, opt.name)
    model_names = [f for f in os.listdir(model_folder) if f.find('%s'%(opt.Snet)) > -1]

    sel_flag = False
    if not opt.best_epoch == -1:                
        idx = [i for i, item in enumerate(model_names) if item.find('%s_%04d' %(opt.Snet, opt.best_epoch)) == 0]
        model_names = [model_names[idx[0]]]        
        sel_flag = True
    
    score_list = list()
    num_models = len(model_names)

    cnt = 0
    for model_name in model_names:        
        
        accelerator.load_state(os.path.join(model_folder, model_name))                

        model_l.eval()
        model_r.eval()
        model_l.to(accelerator.device)
        model_r.to(accelerator.device)

        output_mat = list()
        gt_label_mat = list()

        for index, batch in enumerate(test_dataloader):
            
            inputs = batch["image"].float().to(accelerator.device)
            labels = batch['label'].long().to(accelerator.device)
            name = batch['name'][0]
            
            with torch.no_grad():            
                output = model_l(inputs)
            output = torch.softmax(output, dim=1).cpu().detach()[0].numpy()            

            output = output.transpose(1, 2, 0)        
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)
            image_size = output.shape # it needs to be revised later / SSB            
            
            cimg = denorm_transform(inputs[0])
            cimg = transforms.ToPILImage()(cimg)
            
            gimg = cimg.copy()
            gt_label = labels.cpu().detach()[0].numpy()                        
            g_pal = GetPalletePIL(3, gt_label * 255, image_size[0], image_size[1])
            gimg = Image.blend(gimg.convert("RGBA"), g_pal.convert("RGBA"), alpha=0.5).convert("RGB")

            # Evaluation
            output_mat.append(output.flatten())
            gt_label_mat.append(gt_label.flatten())

            rimg = cimg.copy()        
            o_pal = GetPalletePIL(1, output * 255, image_size[0], image_size[1])
            rimg = Image.blend(rimg.convert("RGBA"), o_pal.convert("RGBA"), alpha=0.5).convert("RGB")

            # Combine images for display
            showimg = Image.fromarray(np.hstack((np.array(cimg), np.array(gimg), np.array(rimg))))
    
            if sel_flag:
                g_pal.save(os.path.join(result_dir, '%s_gt.jpg'%(name)))
                o_pal.save(os.path.join(result_dir, '%s_pred.jpg'%(name)))
                cimg.save(os.path.join(result_dir, '%s_origin.jpg'%(name)))
                showimg.save(os.path.join(result_dir, '%s_showimg.jpg'%(name)))
                    
        
        output_mat = np.array(output_mat)
        gt_label_mat = np.array(gt_label_mat)
        
        cnt += 1
        
        if sel_flag:
            print(model_name)  
            m_IoU_value, IoU_list = compute_mean_iou(output_mat, gt_label_mat)
            f1_value = f1score(output_mat, gt_label_mat)
            score_list.append([str(m_IoU_value), model_name])
            print('crack-IoU: %2.2f %%, back-IoU: %2.2f %%'%(IoU_list[1]*100, IoU_list[0]*100))
            print('m-IoU: %2.2f %%, f1: %2.2f %%'%(m_IoU_value*100, f1_value*100))
        else:                                
            m_IOU, _ = compute_mean_iou(output_mat, gt_label_mat)
            print('[%d / %d] model name: %s, m-IoU: %2.2f %%' % (cnt, num_models, model_name, m_IOU * 100))
            score_list.append([str(m_IOU), model_name])

    if not sel_flag:
        with open(os.path.join(model_folder, '%s_mIoU_history.txt'%(opt.name)), 'w') as f:
            for score in score_list:
                str_score = 'model name: {:s}, m-IoU: {:s}% \n'.format(score[1], score[0])
                f.write(str_score) 

    arr_score = np.array(score_list)
    arr_score = list(map(float, arr_score[:, 0]))
    max_idx = np.argmax(arr_score)
    print(score_list[int(max_idx)])
    print('\n\n\n')



if __name__ == '__main__':
    opt = TestOptions().parse()    
    test_function(opt)
