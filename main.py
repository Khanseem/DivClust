
import random
import numpy as np
import torch
from utils.logger import Logger
from utils.arguments import load_arguments
from data import build_dataset
from torch.utils.data import DataLoader
from engine import build_model, build_optimizer
from engine.trainer import Trainer
import traceback
import os
import resource

def main(args, logger):
    use_cuda = torch.cuda.is_available() and not (isinstance(args.gpu, str) and args.gpu.lower() == "cpu") and not (isinstance(args.gpu, int) and args.gpu < 0)
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    args.device = device
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.print("Loading data")
    dataset_train, dataset_val = build_dataset(args.clustering_framework, args.dataset, args.dataset_path, args)
    train_dataloader = DataLoader(dataset_train, num_workers=args.num_workers, pin_memory=use_cuda, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset_val, num_workers=args.num_workers, pin_memory=use_cuda, batch_size=args.batch_size, shuffle=False, drop_last=False)

    logger.print("Loading model")
    model = build_model(args)
    model.to(device)

    train_steps = len(train_dataloader)*args.epochs
    optimizer = build_optimizer(model, train_steps, args)

    trainer = Trainer(model, optimizer, args, logger)
    
    for ep in range(args.epochs):
        eval = ep%args.eval_interval==0 or ep+1==args.epochs
        trainer.train_epoch(train_dataloader, val_dataloader, eval=eval)

if __name__ == '__main__':
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    args = load_arguments("./configs")
    logger = Logger(args)

    try:
        main(args, logger)
        logger.finish()
    except Exception as e:
        msg = traceback.format_exc()
        f = open(f"{args.output_dir}_error_log.txt", "a")
        f.write(str(msg))
        f.close()
        logger.print(msg)
        logger.finish(crashed=True)
