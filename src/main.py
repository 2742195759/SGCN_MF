from sgcn_mf import SGCN_MF
from parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs

def main():
    """
    Parsing command line parameters, creating target matrix, fitting an SGCN, predicting edge signs, and saving the embedding.
    edge_path
    """
    args = parameter_parser()
    tab_printer(args)
    graph = read_graph(args)

    trainer = SGCN_MF(args, graph)
    trainer.setup_dataset()
    trainer.create_and_train_model()
    if args.test_size > 0:
        trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)

if __name__ == "__main__":
    main()
