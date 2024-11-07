import argparse
import os

import numpy as np
from PIL import Image
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator, DistributedDataParallelKwargs
import random
import torchvision.transforms.functional as TF
from utils.data_load_utils import get_NormTensor, PathFileDataset, func_txtpathfile_read, get_CropNormTensor
from utils.util import get_DenormTensor, pseudo_criterion

from seg_models.network_selector import network_selection
from utils.eval_pre_rec_f1 import compute_mean_iou, f1score
from options.SCPS_train_options import SCPSTrainOptions
# import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torchvision.transforms as transforms




def training_function(opt):
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if opt.with_tracking:
        accelerator = Accelerator(
            cpu=opt.cpu, mixed_precision=opt.mixed_precision, project_dir=opt.name, kwargs_handlers=[ddp_kwargs]
        )
    else:
        accelerator = Accelerator(cpu=opt.cpu, mixed_precision=opt.mixed_precision, kwargs_handlers=[ddp_kwargs])


    model_folder = os.path.join(opt.save_dir, opt.name)
    if not isinstance(opt.image_size, (list, tuple)):
        image_size = (opt.image_size, opt.image_size)
    
    if not isinstance(opt.crop_size, (list, tuple)):
        crop_size = (opt.crop_size, opt.crop_size)


    if opt.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        config = {"lr": opt.lr, "num_epochs": opt.n_epoch, "batch_size": opt.batch_size, "image_size": opt.image_size, "crop_size": opt.crop_size}
        accelerator.init_trackers(run, config)
    
    
    image_tf, label_tf = get_NormTensor()
    denorm_transform = get_DenormTensor(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])    
    
    train_file_names = func_txtpathfile_read(opt.train_dataroot, opt.train_listfile)        
    train_dataset = PathFileDataset(train_file_names, crop_size=crop_size, image_transform=image_tf, label_trainsform=label_tf, phase='train')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=4)
    
    inpaint_file_names = func_txtpathfile_read(opt.inpaint_dataroot, opt.inpaint_listfile)
    inpaint_dataset = PathFileDataset(inpaint_file_names, crop_size=crop_size, image_transform=image_tf, label_trainsform=label_tf, phase='train')
    inpaint_dataloader = DataLoader(inpaint_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=4)                    
    
    valid_file_names = func_txtpathfile_read(opt.val_dataroot, opt.val_listfile)
    valid_dataset = PathFileDataset(valid_file_names, crop_size=image_size, image_transform=image_tf, label_trainsform=label_tf, phase='val')    
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1, num_workers=4)    

    model_l = network_selection(opt)    
    model_r = network_selection(opt)    
    model_l.to(accelerator.device)
    model_r.to(accelerator.device)
    
    scps_criterion = torch.nn.CrossEntropyLoss(reduction='mean')    
    super_criterion = torch.nn.CrossEntropyLoss(reduction='mean')    

    optimizer_l = torch.optim.Adam(params=model_l.parameters(), lr=opt.lr / 5)    
    optimizer_r = torch.optim.Adam(params=model_r.parameters(), lr=opt.lr / 5)    
    lr_scheduler_l = OneCycleLR(optimizer=optimizer_l, max_lr=opt.lr, epochs=opt.n_epoch, steps_per_epoch=len(train_dataloader))
    lr_scheduler_r = OneCycleLR(optimizer=optimizer_r, max_lr=opt.lr, epochs=opt.n_epoch, steps_per_epoch=len(train_dataloader))
    

    model_l, model_r, train_dataloader, inpaint_dataloader, valid_dataloader, lr_scheduler_l, lr_scheduler_r = accelerator.prepare(
        model_l, model_r, train_dataloader, inpaint_dataloader, valid_dataloader, lr_scheduler_l, lr_scheduler_r
    )
    
    overall_step = 0    
    starting_epoch = 0        

    progress_bar = tqdm(
        range(0, opt.n_epoch * train_dataloader.__len__() ),
        initial=overall_step,
        desc="Steps",        
        disable=not accelerator.is_local_main_process,
    )
    score_list = list()
    losses_history = np.zeros((0, 4))
    m_IoU_value = -0.01
    
    iter_inpaint_dataloader = iter(inpaint_dataloader)    
    for epoch in range(starting_epoch, opt.n_epoch):
        model_l.train()
        model_r.train()
        
        iter_train_dataloader = iter(train_dataloader)        
        
        total_loss = 0        
        avg_loss = list()
        avg_loss_sup_l = list()
        avg_loss_sup_r = list()
        avg_cps_loss = list()
  
        for i in range(train_dataloader.__len__()):
                                                           
            s_batch = next(iter_train_dataloader)            
            s_images = s_batch['image'].float().to(accelerator.device)
            s_labels = s_batch['label'].long().to(accelerator.device)                                                      
            
            try:
                u_batch = next(iter_inpaint_dataloader)
            except:
                iter_inpaint_dataloader = iter(inpaint_dataloader)
                u_batch = next(iter_inpaint_dataloader)

            i_images = u_batch['image'].float().to(accelerator.device)
            i_labels = u_batch['label'].long().to(accelerator.device)
            i_names = u_batch['name']
                
            # Input data concatenate btw. super. and unsup.
            in_images = torch.concat([s_images, i_images], dim=0)
            
            pred_l = model_l(in_images)
            pred_r = model_r(in_images)
            
            pred_sup_l = pred_l[:opt.batch_size,]
            pred_sup_r = pred_r[:opt.batch_size,]                    
                                                            
            pred_inpaint_l = pred_l[opt.batch_size:,]
            pred_inpaint_r = pred_r[opt.batch_size:,]
            
            _, max_l = torch.max(pred_inpaint_l, dim=1)
            _, max_r = torch.max(pred_inpaint_r, dim=1)
                                                            
            pseudo_max_l = max_l
            pseudo_max_r = max_r            
            
            
            if (i % opt.lp_freq == 0) and (epoch > opt.lp_epoch):                            
                
                sample_image = denorm_transform(i_images[0,])
                sample_image = transforms.ToPILImage()(sample_image)
                sample_image.save(os.path.join(opt.prog_dir, opt.name, 'Epoch%d_'%(epoch+1) + i_names[0] + '_unsup_orign.jpg'))       
                                
                # save prediction image
                sample_p_image = torch.softmax(pred_inpaint_l[0, ], dim=0)
                sample_p_image = sample_p_image.cpu().detach().numpy().copy()                
                sample_p_image = np.asarray(np.argmax(sample_p_image, axis=0), dtype=np.int32)
                sample_p_image = sample_p_image * 255.0
                sample_p_image = sample_p_image.astype(np.uint8)
                sample_p_image = Image.fromarray(sample_p_image)
                sample_p_image.save(os.path.join(opt.prog_dir, opt.name, 'Epoch%d_'%(epoch+1) + i_names[0] + '_unsup_pred.jpg'))
                
                # save pseudo label                
                sample_p_label = pseudo_max_r[0, ].cpu().detach().numpy().copy()            
                sample_p_label = sample_p_label * 255.0
                sample_p_label = sample_p_label.astype(np.uint8)
                sample_p_label = Image.fromarray(sample_p_label)
                sample_p_label.save(os.path.join(opt.prog_dir, opt.name, 'Epoch%d_'%(epoch+1) + i_names[0] + '_psuedo_label.jpg'))                                                                                                  
            
            cps_loss = pseudo_criterion(scps_criterion, pred_inpaint_l, i_labels, pseudo_max_r, opt.cps_weight) + pseudo_criterion(scps_criterion, pred_inpaint_r, i_labels, pseudo_max_l, opt.cps_weight)                        
            loss_sup_l = super_criterion(pred_sup_l, s_labels)            
            loss_sup_r = super_criterion(pred_sup_r, s_labels)      
            
            sum_loss = loss_sup_l + loss_sup_r + cps_loss  
            accelerator.backward(sum_loss)               
                        
            optimizer_l.step()
            optimizer_r.step()
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            
            if opt.with_tracking:
                total_loss += sum_loss.detach().float()
                
            # accelerator.backward(loss_sup_l + loss_sup_r + cps_loss)
            lr_scheduler_l.step()
            lr_scheduler_r.step()
            overall_step += 1
            progress_bar.update(1)
            avg_loss.append(sum_loss.item())
            
            avg_loss_sup_l.append(loss_sup_l.item())
            avg_loss_sup_r.append(loss_sup_r.item())
            avg_cps_loss.append(cps_loss.item())
            
        avg_loss_sup_l = np.array(avg_loss_sup_l)
        avg_loss_sup_r = np.array(avg_loss_sup_r)
        avg_cps_loss = np.array(avg_cps_loss)

        avg_loss_sup_l = avg_loss_sup_l.mean()
        avg_loss_sup_r = avg_loss_sup_r.mean()
        avg_cps_loss = avg_cps_loss.mean()

        tmp = np.array([epoch, avg_loss_sup_l, avg_loss_sup_r, avg_cps_loss])
        tmp = np.expand_dims(tmp, axis=0)
        losses_history = np.vstack((losses_history, tmp))
        # print(losses_history)
        if opt.with_tracking:
            progress_bar.set_postfix(loss=total_loss.item() / (train_dataloader.__len__()), m_IoU=100 * m_IoU_value)    
        
        ######################################################
        ##################### evaluation #####################
        ######################################################
        
        model_l.eval()
        # model_r.eval()
        output_mat = list()
        gt_label_mat = list()
        
        if ((epoch +1)% opt.save_freq) == 0:
        
            for step, batch in enumerate(valid_dataloader):
                
                inputs = batch["image"].float().to(accelerator.device)
                labels = batch['label'].long().to(accelerator.device)
                
                with torch.no_grad():
                    output = model_l(inputs)
                    
                output = torch.softmax(output, dim=1).cpu().detach()[0].numpy()
                output = output.transpose(1, 2, 0)        
                output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)
                
                gt_label = labels.cpu().detach()[0].numpy()
                
                # Evaluation
                output_mat.append(output.flatten())
                gt_label_mat.append(gt_label.flatten())
            
            output_mat = np.array(output_mat)
            gt_label_mat = np.array(gt_label_mat)
            m_IoU_value, _ = compute_mean_iou(output_mat, gt_label_mat)
            if opt.with_tracking:
                progress_bar.set_postfix(loss=total_loss.item() / (train_dataloader.__len__()), m_IoU=100 * m_IoU_value)    
                
            score_list.append([str(m_IoU_value*100), str(epoch+1)])
        
            
        if ((epoch +1)% opt.save_freq) == 0:            
            model_name = '%s_%04d'%(opt.Snet, epoch+1)
            saved_model = os.path.join(model_folder, model_name)
            accelerator.save_state(saved_model)
                        

    with open(os.path.join(model_folder, '%s_mIoU_history.txt'%(opt.name)), 'w') as f:
        for score in score_list:
            str_score = '{:s}, m-IoU: {:s} \n'.format(score[1], score[0])
            f.write(str_score) 
                    
    # save losses history
    np.save((model_folder + '/%s_losses_history.npy'%(opt.name)), losses_history)        
    
    if opt.with_tracking:
        accelerator.end_training()
            
    


if __name__ == "__main__":
    opt = SCPSTrainOptions().parse()
    training_function(opt)

