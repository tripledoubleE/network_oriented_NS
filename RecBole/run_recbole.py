# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse


from recbole.quick_start import run_recbole
run_recbole(model='LightGCN', config_file_list=['/home/ece/Desktop/Negative_Sampling/lightgcn_parameters.yaml'])


'''

from recbole.quick_start import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    parser.add_argument(
        "--train_neg_sample_args", type=dict, default={"popularity": 1}, help="negative sampling strategy"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="total number of jobs"
    )


    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    run(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
    )

'''
'''
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from recbole.model.general_recommender import *
from recbole.model.context_aware_recommender import *
from recbole.model.knowledge_aware_recommender import *
from recbole.utils.case_study import full_sort_topk, full_sort_scores
import torch
import numpy as np
from recbole.data.interaction import Interaction

from tqdm import tqdm
import json
import argparse
import csv
import pickle

def get_parameter_dict(json_file_path):
    # Open the JSON file in read mode
    with open(json_file_path, 'r') as json_file:
        # Use json.load() to load the JSON data into a dictionary
        parameter_dict = json.load(json_file)
    return parameter_dict

def get_config(model_name, parameter_dict):
    #dataset name is changing according to your dataset name
    config = Config(model= model_name.__name__, dataset='ml-100k', config_dict=parameter_dict)

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    return config, logger, dataset

def dataset_split(config, dataset):
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print("len of valid data " , len(valid_data))
    print("len of train data " , len(train_data))
    print("len of test data " , len(test_data))
    return train_data, valid_data, test_data

def model_training(model_name, config, logger, train_data):
    #TRAINING
    print("Starting Model training")
    print(train_data)
    model = model_name(config, train_data.dataset).to(config['device'])
    print("After model")
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    print("len of train data " , len(train_data))
    
    best_valid_score, best_valid_result = trainer.fit(train_data)
    return model, trainer, best_valid_score, best_valid_result

def evaluation(test_data, trainer):
    test_result = trainer.evaluate(test_data)
    return test_result


def prediction(test_data, dataset, model, config):
    # save all predictions

    prediction_dict = {} 
    # key -> user_id
    # value -> dict: 
                    #key1 -> [full_sort_scores] 
                    #key2 -> [topk_score]
                    #key3 -> [topk_iid_list]
                    #key4 -> [external_item_list]

    for element in tqdm(test_data.dataset.field2id_token['UserId']):
        if element == '[PAD]':
            pass
        else:
            prediction_dict[element] = {}
            uid_series = dataset.token2id(dataset.uid_field, [element])
            score = full_sort_scores(uid_series, model, test_data, device=config['device'])
            #prediction_dict[element]["full_sort_scores"] = score

            topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=5, device=config['device'])
            #prediction_dict[element]["topk_score"] = topk_score
            #prediction_dict[element]["topk_iid_list"] = topk_iid_list

            external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            prediction_dict[element]["external_item_list"] = external_item_list


    return prediction_dict


def main(model_name_str, json_file_path, results_file_path):
    
    model_classes = {
    "ItemKNN": ItemKNN,
    "Pop": Pop,
    "ItemKNN": ItemKNN,
    "BPR": BPR,
    "NeuMF": NeuMF,
    "ConvNCF": ConvNCF,
    "DMF": DMF,
    "FISM": FISM,
    "NAIS": NAIS,
    "SpectralCF": SpectralCF,
    "GCMC": GCMC,
    "NGCF": NGCF,
    "LightGCN": LightGCN,
    "DGCF": DGCF,
    "LINE": LINE,
    "MultiVAE": MultiVAE, 
    "MultiDAE": MultiDAE,
    "MacridVAE": MacridVAE,
    "CDAE": CDAE,
    "ENMF": ENMF,
    "NNCF": NNCF,
    "RaCT": RaCT,
    "RecVAE": RecVAE,
    "EASE": EASE,
    "SLIMElastic": SLIMElastic,
    "SGL": SGL,
    "ADMMSLIM": ADMMSLIM,
    "NCEPLRec": NCEPLRec,
    "SimpleX": SimpleX,
    "NCL": NCL,
    "LR": LR,
    "FM": FM,
    "NFM": NFM,
    "DeepFM": DeepFM,
    "xDeepFM": xDeepFM,
    "AFM": AFM,
    "FFM": FFM,
    "FwFM": FwFM,
    "FNN": FNN,
    "PNN": PNN,
    "DSSM": DSSM,
    "WideDeep": WideDeep,
    #"DIN": DIN,
    #"DIEN": DIEN,
    "DCN": DCN,
    "DCNV2":DCNV2,
    "AutoInt": AutoInt,
    "RippleNet": RippleNet
    }

    model_name = model_classes[model_name_str]

    # get parameters
    parameter_dict = get_parameter_dict(json_file_path)

    # get config
    config, logger, dataset = get_config(model_name, parameter_dict)

    # split data
    train_data, valid_data, test_data = dataset_split(config, dataset)

    # Model training
    
    model, trainer, best_valid_score, best_valid_result = model_training(model_name, config, logger, train_data)
    
    # evaluation
    test_result = evaluation(test_data, trainer)

    evaluation_result_save = results_file_path + "/evaluation_results/" + model_name.__name__ +".csv" 
    
    results_dict =  dict(test_result)
    results_dict["best_valid_score"] = best_valid_score
    results_dict["best_valid_result"] = best_valid_result 
    
    with open(evaluation_result_save, "w") as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(dict(results_dict))
        csvwriter.writerow(dict(results_dict).values())


    
    #In order to save predictions
    prediction_dict = prediction(test_data, dataset, model, config)

    prediction_result_save = results_file_path + "/prediction_results/" + model_name.__name__ +".csv"

    # Flatten the dictionary to create a list of rows
    flattened_data = []
    for user_id, values in prediction_dict.items():
        recomm_items = values['external_item_list'][0] if 'external_item_list' in values else []
        flattened_data.append([user_id, ', '.join(recomm_items)])

    # Open the CSV file in write mode
    with open(prediction_result_save, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(['user_id', 'recomm_items'])
        # Write the data rows
        writer.writerows(flattened_data)
 '''   
'''
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("param1", help="Model Name")
    parser.add_argument("param2", help="Parameter dict path")
    parser.add_argument("param3", help="Results path")
    args = parser.parse_args()
    
    main(args.param1, args.param2, args.param3)

'''