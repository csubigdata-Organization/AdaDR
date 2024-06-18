import argparse
import os
from collections import OrderedDict
import tensorflow as tf
import numpy as np
import sys
import pickle
from loader import data_reader, BatchManager
from supernet import SuperNet
from scratch_model import Model
from utils import create_model
import evaluation
import random


sys.path.append("..")

def parse_args():

    parser = argparse.ArgumentParser(description="Run AdaDR.")
    parser.add_argument('--dataset', nargs='?', default='Fdataset', help='Choose a dataset. [Fdataset/Cdataset/LRSSL]')
    parser.add_argument('--mode', type=str, default='cv', help='cv, case, analysis.')
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument('--gpu', type=str, default="0", help='GPU id.')

    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024*3, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='init Learning rate.')
    parser.add_argument('--disease_dim', type=int, default=125)
    parser.add_argument('--drug_dim', type=int, default=125)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--specific_name", type=str, default='parkinson', help='choose parkinson or breast cancer.')
    parser.add_argument("--specific_id", type=int, default=0)

    parser.add_argument('--arch_lr', type=float, default=0.1, help='init Learning rate for arch search')
    parser.add_argument('--alpha_mode', type=str, default='train', choices=['train', 'valid'], help='update architecture alpha based on train/valid data')
    parser.add_argument('--num_sampled_archs', type=int, default=5, help='sample archs from supernet')


    return parser.parse_args()

args = parse_args()

specific_id = 0
if args.mode == "case":
    args.seed = 11
    if args.specific_name == "parkinson":
        specific_id = 119
    elif args.specific_name == "breast cancer":
        specific_id = 19

def config_model():
    config = OrderedDict()
    config['dataset'] = args.dataset
    config['lr'] = args.lr
    config['batch_size'] = args.batch_size
    config['disease_dim'] = args.disease_dim
    config['drug_dim'] = args.drug_dim
    config['hidden_dim'] = args.hidden_dim
    config["clip"] = 3
    config['log_path'] = "log/" + args.dataset
    config['max_epoch'] = args.epochs
    config['mode'] = args.mode
    config['seed'] = args.seed
    config['scratch_seed'] = args.seed
    config['n_splits'] = args.n_splits
    config['specific_name'] = args.specific_name
    config['specific_id'] = specific_id
    config['gpu'] = args.gpu

    config['arch_lr'] = args.arch_lr
    config['alpha_mode'] = args.alpha_mode
    config['num_sampled_archs'] = args.num_sampled_archs
    return config

config = config_model()

os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']




def evaluate(sess, model, name, alpha_mode, data):
    print("evaluate data:{}".format(name))
    scores, labels = [], []
    for batch in data.iter_batch():
        score, label = model.run_step(sess, name, alpha_mode, batch)
        scores.append(score)
        labels.append(label)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    result = evaluation.evaluate(scores, labels)
    auroc = result['auroc']
    aupr = result['aupr']
    return auroc, aupr


def evaluate_scratch(sess, model, data):
    scores, labels = [], []
    for batch in data.iter_batch():
        score, label = model.run_step(sess, False, batch)
        scores.append(score)
        labels.append(label)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    result = evaluation.evaluate(scores, labels)
    auroc = result['auroc']
    aupr = result['aupr']
    return auroc, aupr



def search():
    map_file_path = "./pkl/" + config['dataset'] + "/data.pkl"
    if os.path.isfile(map_file_path):
        with open(map_file_path, "rb") as f:
            disease_disease_sim_Matrix, drug_drug_sim_Matrix, truth_label, \
            all_train_mask, all_test_mask = pickle.load(f)
    else:
        disease_disease_sim_Matrix, drug_drug_sim_Matrix, truth_label, \
        all_train_mask, all_test_mask = data_reader(config=config,
                                                    dataset=config['dataset'])
    random.seed(config['seed'])

    tf.reset_default_graph()
    tf.set_random_seed(config['seed'])
    np.random.seed(config['seed'])

    search_fold = 0 # search_fold can be any fold, such as 0, 5, -1
    train_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_train_mask[search_fold], truth_label)
    valid_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[search_fold], truth_label)
    test_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[search_fold], truth_label)
    train_manager = BatchManager(train_data, config['batch_size'], "train")
    valid_manager = BatchManager(valid_data, config['batch_size'], 'valid')
    test_manager = BatchManager(test_data, config['batch_size'], "test")
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data


    with tf.Session(config=tf_config) as sess:
        ckptpath = "ckpt/{}/{}-search/".format(config['dataset'], config['dataset'])
        supernet = create_model(sess, SuperNet, ckptpath, config)
        print("start searching")
        loss_list = []
        for i in range(config['max_epoch']):
            for batch in train_manager.iter_batch(shuffle=True):
                step, loss, z = supernet.run_step(sess, 'train', config['alpha_mode'], batch)
                loss_list.append(loss)
                if step % 20 == 0:
                    iteration = step // steps_per_epoch + 1
                    print("epoch:{}, step:{}/{}, loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss_list)))
                    loss_list = []

            train_manager = BatchManager(train_data, config['batch_size'], "train")

            auroc, aupr = evaluate(sess, supernet, "valid", config['alpha_mode'], valid_manager)
            print("epoch:{}, valid auroc :{:>.5f}, valid aupr :{:>.5f}".format(iteration, auroc, aupr))


            searched_architecture = supernet.get_best_architecture_alpha(sess)
            print('seed:{}, epoch:{}, searched_architecture={}'.format(config['seed'], iteration, searched_architecture))


    return searched_architecture



def train_scratch(architecture):
    config['scratch_architecture'] = architecture

    map_file_path = "./pkl/" + config['dataset'] + "/data.pkl"
    if os.path.isfile(map_file_path):
        with open(map_file_path, "rb") as f:
            disease_disease_sim_Matrix, drug_drug_sim_Matrix, truth_label, \
            all_train_mask, all_test_mask = pickle.load(f)
    else:
        disease_disease_sim_Matrix, drug_drug_sim_Matrix, truth_label, \
        all_train_mask, all_test_mask = data_reader(config=config,
                                                    dataset=config['dataset'])
    final_all_auroc, final_all_aupr = [], []
    random.seed(config['seed'])
    for fold_num in range(len(all_train_mask)):

        tf.reset_default_graph()
        tf.set_random_seed(config['seed'])
        np.random.seed(config['seed'])

        train_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_train_mask[fold_num], truth_label)
        valid_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[fold_num], truth_label)
        test_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[fold_num], truth_label)
        train_manager = BatchManager(train_data, config['batch_size'], "train")
        valid_manager = BatchManager(valid_data, config['batch_size'], 'valid')
        test_manager = BatchManager(test_data, config['batch_size'], "test")
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_data
        with tf.Session(config=tf_config) as sess:
            ckptpath = "ckpt/{}/{}-scratch-fold{}/".format(config['dataset'], config['dataset'], fold_num + 1)
            model = create_model(sess, Model, ckptpath, config)
            print("start training fold {}".format(fold_num + 1))
            loss_list = []
            for i in range(config['max_epoch']):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, loss, z = model.run_step(sess, True, batch)
                    loss_list.append(loss)
                    if step % 20 == 0:
                        iteration = step // steps_per_epoch + 1
                        print("epoch:{} step:{}/{}, loss:{:>9.6f}".format(
                            iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss_list)))
                        loss_list = []
                train_manager = BatchManager(train_data, config['batch_size'], "train")
                auroc, aupr = evaluate_scratch(sess, model, valid_manager)
                print("fold {} valid auroc :{:>.5f}".format(fold_num + 1, auroc))
                print("fold {} valid aupr :{:>.5f}".format(fold_num + 1, aupr))
            final_test_auroc, final_test_aupr = evaluate_scratch(sess, model, test_manager)
            final_all_auroc.append(final_test_auroc)
            final_all_aupr.append(final_test_aupr)
            print("fold {} final test auroc :{:>.5f}".format(fold_num + 1, final_test_auroc))
            print("fold {} final test aupr :{:>.5f}".format(fold_num + 1, final_test_aupr))
    print("final_avg_auroc :{:>.5f} final_avg_aupr :{:>.5f}".format(np.mean(final_all_auroc),
                                                                    np.mean(final_all_aupr)))
    return np.mean(final_all_auroc), np.std(final_all_auroc), np.mean(final_all_aupr), np.std(final_all_aupr)


def run_by_seed():
    searched_architectures = []
    for i in range(config['num_sampled_archs']):
        print('searched {}-th for {}...'.format(i+1, config['dataset']))
        seed = np.random.randint(0, 10000)
        config['seed'] = seed

        searched_architecture = search()
        searched_architectures.append(searched_architecture)


    ## train from scratch
    config['seed'] = config['scratch_seed'] # keep same as the previous work
    auroc, aupr, achitecture = 0, 0, None
    auroc_std, aupr_std = 0, 0
    for i, searched_architecture in enumerate(searched_architectures):
        print('train from scratch {}-th searched architecture ({}) for {}...'.format(i + 1, searched_architecture, config['dataset']))

        searched_architecture = searched_architecture.split('||')


        temp_auroc, temp_auroc_std, temp_aupr, temp_aupr_std = train_scratch(searched_architecture)
        if temp_aupr >= aupr:
            aupr = temp_aupr
            aupr_std = temp_aupr_std
            auroc = temp_auroc
            auroc_std = temp_auroc_std
            achitecture = '||'.join(searched_architecture)
    print('train from scratch ending')
    print('*'*50)
    print('The final search results for {} as follows:'.format(config['dataset']))
    print('\t\t\t auroc:{:>.5f}, auroc_std:{:>.3f}'.format(auroc, auroc_std))
    print('\t\t\t aupr:{:>.5f}, aupr_std:{:>.3f}'.format(aupr, aupr_std))
    print('\t\t\t achitecture:{}'.format(achitecture))
    print('*' * 50)



if __name__ == "__main__":


    run_by_seed()

