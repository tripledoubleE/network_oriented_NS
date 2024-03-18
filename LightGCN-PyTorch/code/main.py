import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from tqdm import tqdm

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in tqdm(range(world.TRAIN_epochs)):
        start = time.time()
        cprint("[TEST]")
        Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        # if epoch %10 == 0:
        #     cprint("[TEST]")
        #     Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        if world.config['neg_sample'] == 'uniform':
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        if world.config['neg_sample'] == 'alpha75':
            output_information = Procedure.Alpha75_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        elif world.config['neg_sample'] == 'mcns':
            output_information = Procedure.MCNS_train(dataset, Recmodel, bpr, epoch, w=w)
        elif world.config['neg_sample'] == 'item_proj':
            output_information = Procedure.ItemProj_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        elif world.config['neg_sample'] == 'user_proj':
            output_information = Procedure.UserProj_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        elif world.config['neg_sample'] == 'dens':
            output_information = Procedure.Dens_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        elif world.config['neg_sample'] == 'dynamic':
            output_information = Procedure.Dynamic_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        elif world.config['neg_sample'] == 'item_proj_SimRank':
            output_information = Procedure.SimRank_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        elif world.config['neg_sample'] == 'item_proj_Panther':
            output_information = Procedure.Panther_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        elif world.config['neg_sample'] == 'metapath2vec':
            output_information = Procedure.MetaPath2Vec_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        elif world.config['neg_sample'] == 'no_sample': 
            output_information = Procedure.NoSampling_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)       
        elif world.config['neg_sample'] == 'naive_random_walk': 
            output_information = Procedure.Naive_random_walk_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)



        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()