from sgcn_mf import SGCN_MF
from parser import parameter_parser
from utils import tab_printer, read_dataset_split_bytime, score_printer, save_logs , build_graph , XKLOG
import torch
import numpy
import random
from trainer import Trainer

def main():

    args = parameter_parser()
    tab_printer(args)
    log = XKLOG('./log/')
    log.setpara(args) 
    args.log = log
    # make the random get the same output 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    random.seed(args.seed)

    trainset , testset = read_dataset_split_bytime(args) # split the dataset and share a encoder for all str
    traingraph = build_graph(args , trainset)            # build the graph from the dataset
    testgraph = build_graph(args , testset)

    if args.model == 'sgcn_mf' : 
        log.setdescri('sgcn_mf'+'_'+args.description)
        trainer = SGCN_MF(args, traingraph , testgraph)
        trainer.train()

    elif args.model == 'mf' : 
        log.setdescri('mf'+'_'+args.description)
        t = Trainer(args ,traingraph , testgraph)
        t.train()

    log.write()
        
        

if __name__ == "__main__":
    main()
