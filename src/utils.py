import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing

def unique_2dim_list(li , idxs) : 
    s = set()
    res = []
    for i in li : 
        lidx = []
        for idx in idxs : 
            lidx.append(i[idx])
        if tuple(lidx) not in s :
            s.add(tuple(lidx)) 
            res.append(i)
    return res 

def gather_2dim_list(li , idx) : 
    res = {}
    for i in li : 
        key = i[idx]
        if key not in res : res[key] = []
        res[key].append(i)
    return res

def idconvert(args , idx) : 
    nu = args.encoder['nu']
    ni = args.encoder['ni'] 
    nf = args.encoder['nf'] 
    if idx < nu : return idx
    if idx < ni : return idx-nu
    if idx < nf : return idx-nu-ni
    
def build_edge_from_hypergraph(args , hyper_edge) : 
    nu = args.encoder['nu']
    ni = args.encoder['ni'] 
    nf = args.encoder['nf'] 
    pos = [] 
    neg = []
    for edge in hyper_edge : 
        if edge[3] == 1 : 
            pos.append([edge[0] , nu + edge[1]])
            pos.append([edge[0] , nu+ni+edge[2]])
            pos.append([nu+edge[1] , nu+ni+edge[2]])
        elif edge[3] == -1 : 
            neg.append([edge[0] , nu + edge[1]])
            neg.append([edge[0] , nu+ni+edge[2]])
            neg.append([nu+edge[1] , nu+ni+edge[2]])
    return pos , neg

'''
auc , mrr , preci , recall , f1 , ndcg , hitratio
f1
'''
def split_by_user_time (args , dataset , userid , timeidx) : 
    tmp = gather_2dim_list(dataset , userid)
    train = []
    test = []
    test_train_ratio = args.test_size
    for li in tmp.values() : 
        sor = sorted(li , key=lambda x : x[timeidx])
        numtest = int(len(sor)*test_train_ratio)
        numtest = numtest if numtest != 0 else 1
        train.extend(sor[0:numtest])
        test.extend(sor[numtest:])
    return train , test

def read_dataset_split_bytime(args):
    dataset = pd.read_csv(args.data_path , sep='\t' , header=None).values.tolist()
    ## change name to id
    enc_user = preprocessing.LabelEncoder()
    enc_item = preprocessing.LabelEncoder()
    enc_feature = preprocessing.LabelEncoder()
    enc_user.fit([edge[0] for edge in dataset])
    enc_item.fit([edge[1] for edge in dataset])
    feature_labels = []
    for features in [edge[3] for edge in dataset] : 
        for fs in features.split(':') : 
            feature_labels.append(fs.split('|')[0])
    enc_feature.fit(feature_labels)
    encoder = {}
    encoder['nu'] = len(enc_user.classes_)
    encoder['ni'] = len(enc_item.classes_)
    encoder['nf'] = len(enc_feature.classes_)
    encoder['enc_user'] = enc_user
    encoder['enc_item'] = enc_item
    encoder['enc_feature'] = enc_feature
    train , test = split_by_user_time(args , dataset , 0 ,  4)
    args.encoder = encoder
    return train , test

def build_graph(args , dataset) : 
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    encoder = args.encoder
    enc_user = encoder['enc_user'] 
    enc_item = encoder['enc_item'] 
    enc_feature = encoder['enc_feature'] 
    nu = args.encoder['nu']
    ni = args.encoder['ni'] 
    nf = args.encoder['nf'] 
    graph = {}
    hyper_edge = []
    for line in dataset : 
        for features in line[3].split(':') : 
            ff = features.split('|')
            hyper_edge.append([line[0] , line[1] , ff[0] , ff[2] , line[2]])

    hyper_edge = [[enc_user.transform([edge[0]])[0] , enc_item.transform([edge[1]])[0] , enc_feature.transform([edge[2]])[0] , int(edge[3]) , int(edge[4])] for edge in hyper_edge]
    interaction = [[int(edge[0]) , int(edge[1]) , int(edge[4])] for edge in unique_2dim_list(hyper_edge , [0,1])]
        
    pos , neg = build_edge_from_hypergraph(args , hyper_edge)

    hyper_edge = [ [edge[0] , edge[1]+nu , edge[2]+nu+ni , edge[3]] for edge in hyper_edge ]  

    graph["positive_edges"] = pos
    graph["negative_edges"] = neg
    graph["ecount"] = len(pos) + len(neg)
    graph["ncount"] = encoder['nu']+encoder['ni']+encoder['nf']
    graph["interaction"] = interaction # userid , itemid , rating  [seperate_id]
    graph["hyper_edge"] = hyper_edge   # userid , itemid , featureid , +/-1 [gather_id]

    return graph 

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    for key in keys : 
        print (key , '\t\t' , args[key])

def calculate_auc(targets, predictions, edges):
    """
    Calculate performance measures on test dataset.
    :param targets: Target vector to predict.
    :param predictions: Predictions vector. 
    :param edges: Edges dictionary with number of edges etc.
    :return auc: AUC value.
    :return f1: F1-score.
    """
    neg_ratio = len(edges["negative_edges"])/edges["ecount"]
    targets = [0 if target == 1 else 1 for target in targets]
    auc = roc_auc_score(targets, predictions)
    f1 = f1_score(targets, [1 if p > neg_ratio else 0 for p in  predictions])
    return auc, f1

def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """
    t = Texttable() 
    t.add_rows([per for i, per in enumerate(logs["performance"]) if i % 10 == 0])
    print(t.draw())

def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    with open(args.log_path,"w") as f:
            json.dump(logs,f)

def setup_features(args, positive_edges, negative_edges, node_count):
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    if args.spectral_features:
        X = create_spectral_features(args, positive_edges, negative_edges, node_count)
    else:
        X = create_general_features(args)
    return X

def create_general_features(args):
    """
    Reading features using the path.
    :param args: Arguments object.
    :return X: Node features.
    """
    X = np.array(pd.read_csv(args.features_path))
    return X

def create_spectral_features(args, positive_edges, negative_edges, node_count):
    """
    Creating spectral node features using the train dataset edges.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    pass
