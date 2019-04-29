from sgcn_mf import SGCN_MF
from parser import parameter_parser
from utils import tab_printer, read_dataset_split_bytime, score_printer, save_logs , build_graph
import torch
from trainer import Trainer

def main():
    """
    Parsing command line parameters, creating target matrix, fitting an SGCN, predicting edge signs, and saving the embedding.
    edge_path
    """
    args = parameter_parser()
    tab_printer(args)

    trainset , testset = read_dataset_split_bytime(args) # split the dataset and share a encoder for all str
    traingraph = build_graph(args , trainset)            # build the graph from the dataset
    testgraph = build_graph(args , testset)

    if args.model == 'sgcn_mf' : 
        trainer = SGCN_MF(args, traingraph , testgraph)
        trainer.train()
        if args.test_size > 0:
            trainer.save_model()
            score_printer(trainer.logs)
            save_logs(args, trainer.logs)

    elif args.model == 'mf' : 
        t = Trainer(args ,traingraph , testgraph)
        t.train()
        
        

if __name__ == "__main__":
    main()
