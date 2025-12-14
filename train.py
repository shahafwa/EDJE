'''
 * Simple Efficient Fusion - Multimodal Framework
 * Licensed under BSD 3-Clause License
 * Anonymous submission for review
'''
import argparse
import time

import numpy as np
import random
import datetime
import json
from pathlib import Path
from ruamel.yaml import YAML
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from transformers import AutoImageProcessor
from dotenv import load_dotenv
import os

from data.flickr30k_dataset import flickr30k_retrieval_eval
from data.coco_karpathy_dataset import coco_karpathy_retrieval_eval
from evaluation.evaluate_retrieval import evaluation_t2i, evaluation_i2t
from exp_logger import ExperimentLogger
from pretrain_model import MultimodalPretrainModel
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader


torch.set_default_dtype(torch.bfloat16)

# Load environment variables from .env file
load_dotenv()

def train(model, teacher_model, data_loader, optimizer, epoch, device, config, logger=None,
          global_step=0, model_without_ddp=None, output_dir=None):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_distil', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_checkpoint_freq = 1_000  # Save checkpoint every 1000 steps

    data_loader.sampler.set_epoch(epoch)

    for i, (cur_data) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        current_step = global_step + i
        if len(cur_data) == 3:
            image, caption, img_ids = cur_data
        else:
            image, caption = cur_data
            img_ids = None
            
        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
            
        optimizer.zero_grad()
        
        image = image.to(device, non_blocking=True)
        
        # Forward pass with optional negative images
        model_outputs = model(image, caption, img_ids=img_ids, teacher_model=teacher_model)
        loss_lm, loss_itc, loss_itm, loss_distil = model_outputs
        loss = loss_lm + loss_itc + loss_itm + loss_distil
            
        loss.backward()
        optimizer.step()    

        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_distil=loss_distil.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Save step-based checkpoint every 1000 steps (only on main process)
        if (current_step > 0 and current_step % step_checkpoint_freq == 0 and
            utils.is_main_process() and model_without_ddp is not None and output_dir is not None):
            print(f"Saving step checkpoint at step {current_step}")

            # Use a fixed name for step checkpoints (overwrites previous)
            step_checkpoint_path = os.path.join(output_dir, 'checkpoint_latest_step.pth')

            # Delete previous step checkpoint if it exists
            if os.path.exists(step_checkpoint_path):
                try:
                    os.remove(step_checkpoint_path)
                    print(f"Removed previous step checkpoint: {step_checkpoint_path}")
                except OSError as e:
                    print(f"Warning: Could not remove previous step checkpoint: {e}")

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'global_step': current_step,
            }
            torch.save(save_obj, step_checkpoint_path)
            print(f"Saved step checkpoint at step {current_step} to {step_checkpoint_path}")

            # Log step checkpoint to experiment logger if available
            if logger:
                logger.log_checkpoint(step_checkpoint_path, epoch=epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    
    # Log training metrics to experiment logger (only on main process)
    if logger and utils.is_main_process():
        for metric_name, meter in metric_logger.meters.items():
            logger.log_metric(f'train_{metric_name}', meter.global_avg, step=epoch)

    # Return both training stats and the number of steps completed in this epoch
    steps_completed = len(data_loader)
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, steps_completed


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Only initialize experiment logger on main process to avoid duplicate experiments
    exp_logger = None
    if utils.is_main_process():
        print(f"Initializing experiment logger on main process (rank {utils.get_rank()})")
        exp_logger = ExperimentLogger(config)
    else:
        print(f"Skipping experiment logger initialization on worker process (rank {utils.get_rank()})")
    #### Dataset ####
    print("Creating dataset")
    creator = create_dataset
    datasets = [creator('retrieval', config)]
    image_processor = AutoImageProcessor.from_pretrained(config['siglip_path'])
    siglip_transform = lambda x: image_processor(images=x, return_tensors='pt')['pixel_values'][0]
    
    # Choose test dataset based on config
    test_set = config.get('test_set', 'flickr')  # Default to flickr if not specified
    
    # Handle both single string and list of test sets
    if isinstance(test_set, str):
        test_set = [test_set]
    
    coco_test_dataset = None
    flickr_test_dataset = None
    
    for dataset_name in test_set:
        if 'coco' in dataset_name:
            coco_test_dataset = coco_karpathy_retrieval_eval(siglip_transform, config['coco_image_root'], config['coco_ann_root'], 'test')
            print("Using COCO test dataset for evaluation")
        if 'flickr' in dataset_name:
            flickr_test_dataset = flickr30k_retrieval_eval(siglip_transform, config['flickr_image_root'], config['flickr_ann_root'], 'test')
            print("Using Flickr30k test dataset for evaluation")
    print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()            
    samplers = create_sampler(datasets, [not (config['data'] in ["imagenet", "imagenet_zeroshot", "mixture", "laion"])],
                              num_tasks, global_rank)
    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]

    #### Model #### 
    print("Creating model")
    model = MultimodalPretrainModel(
        siglip_path=config['siglip_path'],
        base_language_model_path=config['language_model_path'],
        multimodal_projection_hidden_dim=config['multimodal_projection_hidden_dim'],
        num_negatives_per_sample=config.get('num_negatives_per_sample', 1),
        finetune=config.get('finetune', False),
        num_compressed_tokens=config.get('num_compressed_tokens', None)
    )

    model = model.to(device)

    teacher_model = MultimodalPretrainModel(
        siglip_path=config['teacher_siglip_path'],
        base_language_model_path=config['language_model_path'],
        multimodal_projection_hidden_dim=config['multimodal_projection_hidden_dim'],
        num_negatives_per_sample=config['num_negatives_per_sample'],
    )
    teacher_model_sd = torch.load(args.teacher_checkpoint)["model"]
    teacher_model.load_state_dict(teacher_model_sd)
    teacher_model = teacher_model.to(device).eval()


    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    start_epoch = 0
    initial_global_step = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # Do not load ANY vision backbone weights from checkpoint. Keep only non-vision modules
        # like language model, projections, classification heads, etc.
        keys_to_remove = []
        for key in list(state_dict.keys()):
            # Remove SigLIP vision encoder weights
            if key.startswith('vision_encoder.'):
                keys_to_remove.append(key)
                continue
            # Remove entire CLIP/SigLIP model weights from checkpoint (we rely on fresh pretrained)
            if key.startswith('clip.'):
                keys_to_remove.append(key)
                continue
        if keys_to_remove:
            print(f"Removing {len(keys_to_remove)} vision-related keys from checkpoint state_dict")
            for key in keys_to_remove:
                del state_dict[key]
            
        # Remove any parameters with shape mismatches (e.g., extended positional embeddings)
        model_state = model.state_dict()
        mismatch_keys = []
        for key in list(state_dict.keys()):
            if key in model_state and state_dict[key].shape != model_state[key].shape:
                mismatch_keys.append((key, tuple(state_dict[key].shape), tuple(model_state[key].shape)))
        if mismatch_keys:
            print(f"Skipping {len(mismatch_keys)} keys due to shape mismatch:")
            for key, old_shape, new_shape in mismatch_keys:
                print(f" - {key}: ckpt {old_shape} -> model {new_shape}")
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

        def _prune_mismatched_optimizer_state(opt: torch.optim.Optimizer):
            # Remove any state tensors that don't match current parameter shapes
            num_pruned = 0
            for group in opt.param_groups:
                for p in group['params']:
                    st = opt.state.get(p, None)
                    if not st:
                        continue
                    pruned_this = False
                    for key in list(st.keys()):
                        val = st[key]
                        if torch.is_tensor(val) and val.shape != p.shape:
                            pruned_this = True
                            break
                    if pruned_this:
                        opt.state[p] = {}
                        num_pruned += 1
            if num_pruned:
                print(f'Pruned optimizer state for {num_pruned} parameter(s) due to shape mismatch')

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            _prune_mismatched_optimizer_state(optimizer)
            print('Successfully loaded optimizer state from checkpoint (after pruning mismatches if any)')
        except Exception as e:
            print(f'Could not load optimizer state from checkpoint: {e}')
            print('Optimizer will start with fresh state (this is normal when model architecture changes)')
        start_epoch = checkpoint['epoch']+1
        print('resume checkpoint from %s'%args.checkpoint)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module



    print("Start training")
    start_time = time.time()
    global_step = initial_global_step  # Initialize global step counter from checkpoint or 0

    for epoch in range(start_epoch, config['max_epoch']):

        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])

        train_stats, steps_completed = train(model, teacher_model, data_loader, optimizer, epoch, device, config, exp_logger,
                                           global_step=global_step, model_without_ddp=model_without_ddp, output_dir=args.output_dir)

        # Update global step counter
        global_step += steps_completed

        # Set model to evaluation mode before evaluation (all processes)
        model_without_ddp.eval()

        if utils.is_main_process():

            # Initialize metrics dictionaries
            all_metrics_t2i = {}
            all_metrics_i2t = {}

            # Check if any test datasets are available
            if not flickr_test_dataset and not coco_test_dataset:
                print("Warning: No test datasets available for evaluation")

            # Evaluate on Flickr30k if available
            if flickr_test_dataset:
                print("=== Evaluating Flickr30k Text-to-Image Retrieval ===")
                flickr_metrics_t2i = evaluation_t2i(model_without_ddp, flickr_test_dataset, device, config)
                print("=== Evaluating Flickr30k Image-to-Text Retrieval ===")
                flickr_metrics_i2t = evaluation_i2t(model_without_ddp, flickr_test_dataset, device, config)
                print(f"Flickr30k T2I Results: {flickr_metrics_t2i}")
                print(f"Flickr30k I2T Results: {flickr_metrics_i2t}")

                # Add flickr metrics with prefix
                all_metrics_t2i.update({f'flickr_{k}': v for k, v in flickr_metrics_t2i.items()})
                all_metrics_i2t.update({f'flickr_{k}': v for k, v in flickr_metrics_i2t.items()})

            # Evaluate on COCO if available
            if coco_test_dataset:
                print("=== Evaluating COCO Text-to-Image Retrieval ===")
                coco_metrics_t2i = evaluation_t2i(model_without_ddp, coco_test_dataset, device, config)
                print("=== Evaluating COCO Image-to-Text Retrieval ===")
                coco_metrics_i2t = evaluation_i2t(model_without_ddp, coco_test_dataset, device, config)
                print(f"COCO T2I Results: {coco_metrics_t2i}")
                print(f"COCO I2T Results: {coco_metrics_i2t}")

                # Add coco metrics with prefix
                all_metrics_t2i.update({f'coco_{k}': v for k, v in coco_metrics_t2i.items()})
                all_metrics_i2t.update({f'coco_{k}': v for k, v in coco_metrics_i2t.items()})

            # Set model back to training mode for next epoch
            model_without_ddp.train()

            # Log evaluation metrics to experiment logger
            if exp_logger:
                for metric, score in all_metrics_t2i.items():
                    exp_logger.log_metric(f't2i_eval_{metric}', score, step=epoch)
                for metric, score in all_metrics_i2t.items():
                    exp_logger.log_metric(f'i2t_eval_{metric}', score, step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f't2i_eval_{metric}': score for metric, score in all_metrics_t2i.items()},
                         **{f'i2t_eval_{metric}': score for metric, score in all_metrics_i2t.items()},
                         'epoch': epoch,
                        }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
            }
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch)
            torch.save(save_obj, checkpoint_path)

            # Log checkpoint path and epoch to Comet (metrics already logged above)
            if exp_logger:
                exp_logger.log_checkpoint(checkpoint_path, epoch=epoch)

            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # End experiment logging
    if utils.is_main_process() and exp_logger:
        exp_logger.log_metric('total_training_time_seconds', total_time)
        exp_logger.end()
        print("Experiment logging ended successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval.yaml')
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--teacher_checkpoint', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    config_yaml = YAML(typ="safe")
    config = config_yaml.load(open(args.config, 'r'))
    config['finetune'] = args.finetune
    if args.finetune:
        config['init_lr'] = config['finetune_lr']
        config['train_file'] = config['finetune_files']
    else:
        config['init_lr'] = config['pretraining_lr']
        config['train_file'] = config['pretrain_files']

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)
