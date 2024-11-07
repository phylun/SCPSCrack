#! /bin/bash
accelerate launch --multi_gpu --main_process_port=29559 accelerate_SelfCPS_SegTrain.py --name CPSInpaintCrack_PoolFormerS36_SD21 --mixed_precision 'bf16' --Snet 'PoolformerS36' --inpainting --with_tracking
