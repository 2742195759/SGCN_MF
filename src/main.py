from sgcn_mf import SGCN_MF
from parser import parameter_parser
from utils import tab_printer, read_dataset_split_bytime, score_printer, save_logs , build_graph

def main():
    """
    Parsing command line parameters, creating target matrix, fitting an SGCN, predicting edge signs, and saving the embedding.
    edge_path
    """
    args = parameter_parser()
    tab_printer(args)
    trainset , testset = read_dataset_split_bytime(args)
    traingraph = build_graph(args , trainset)
    testgraph = build_graph(args , testset)

    trainer = SGCN_MF(args, traingraph , testgraph)
    trainer.setup_dataset()
    trainer.create_and_train_model()
    if args.test_size > 0:
        trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)

if __name__ == "__main__":
    main()
