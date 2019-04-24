from sgcn_mf import SGCN_MF
from MF import MF
from parser import parameter_parser
from utils import tab_printer, read_dataset_split_bytime, score_printer, save_logs , build_graph
from tqdm import trange
import torch

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
        trainer.setup_dataset()
        trainer.create_and_train_model()
        if args.test_size > 0:
            trainer.save_model()
            score_printer(trainer.logs)
            save_logs(args, trainer.logs)

    elif args.model == 'mf' : 
        model = MF(args , args.encoder['nu'] , args.encoder['ni'])
        epochs = trange(args.epochs, desc="Loss")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.mf_learnrate, weight_decay=args.weight_decay) 
        for epoch in epochs : 
            loss = model(torch.LongTensor(traingraph['interaction']))
            loss.backward()
            optimizer.step()
            epochs.set_description("SGCN (Loss=%g)" % round(loss.item(),4))
            if args.test_size >0:
                print (model.score(traingraph, testgraph))
        

if __name__ == "__main__":
    main()
