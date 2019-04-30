import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import time


class XKLOG(object) : 
    def __init__(self , prefix , descri=None , parameters=None) : 
        assert(isinstance(prefix , str))
        self.setdescri(descri)
        self.prefix = prefix
        self.setpara(parameters)
        self.output = []

    def setpara(self , parameters) : 
        self.parameters = parameters
        self.parameters = self.para2str()

    def setdescri(self,descri) : 
        if (isinstance(descri , str)) : 
            self.descri = descri
        else : 
            print ('[WARN] descri not set')

    def para2str(self) : 
        data = ''
        if self.parameters != None : 
            args = vars(self.parameters)
            keys = sorted(args.keys())
            for key in keys : 
                data += str(key) + '\t' + str(args[key]) + '\n'
        return data
    
    def LOG(self , *message) : 
        if message == None or len(message) == 0 : return
        print ('[XKLOG]' , *message)
        self.output.append(message)

    def get_filename(self) : 
        return self.prefix + self.descri + time.strftime('%Y-%m-%d_%H:%M:%S' , time.localtime(time.time())) 

    def write(self ) : 
        with open(self.get_filename() , 'w') as fp : 
            fp.write(self.parameters)
            fp.write("####################\n")
            for out in self.output : 
                line = ''
                if isinstance(out , tuple) : 
                    for item in out : 
                        line += str(item) + ' '
                else : 
                    line += str(out)
                fp.write(line+'\n')
        self.output = []
        
    def testcase() : 
        log = XKLOG('./' , 'test' , None)
        log.LOG(1,2,3,4)
        log.LOG('xiongkun')
        log.LOG([1,2,3,4])
        print (log.get_filename())
        log.write()


class Hasher(object) : 
    def __init__(self , li=None) : 
        self.tr = {}
        self.inv = {}
        if li != None : 
            self.feed(li)

    def feed(self , li) : 
        cnt = 0
        for name in li : 
            if name not in self.tr : 
                self.tr[name] = cnt 
                self.inv[cnt] = name
                cnt += 1

    def tran(self , name) : 
        return self.tr[name]

    def invt(self , idx) : 
        return self.inv[idx]

    def testcase() : 
        h = Hasher(['name' , 'xk' , 'wt' , 'xk'])
        assert(h.tran('xk') == 1)
        assert(h.tran('name') == 0)
        assert(h.tran('wt') == 2)
        assert(h.invt(2)=='wt')

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
        #numtest = numtest if numtest != 0 else 1
        numtrain = len(sor) - numtest
        train.extend(sor[0:numtrain])
        test.extend(sor[numtrain:])
    print (len(train) , len(test))
    return train , test

def read_dataset_split_bytime(args):
    dataset = pd.read_csv(args.data_path , sep='\t' , header=None).values.tolist()
    ## change name to id
    enc_user = Hasher()
    enc_item = Hasher()
    enc_feature = Hasher()
    enc_user.feed([edge[0] for edge in dataset])
    enc_item.feed([edge[1] for edge in dataset])
    feature_labels = []
    for features in [edge[3] for edge in dataset] : 
        for fs in features.split(':') : 
            feature_labels.append(fs.split('|')[0])
    enc_feature.feed(feature_labels)
    encoder = {}
    encoder['nu'] = len(enc_user.tr)
    encoder['ni'] = len(enc_item.tr)
    encoder['nf'] = len(enc_feature.tr)
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

    

    print("Collected hyper_edge" , len(hyper_edge))
    hyper_edge = [[enc_user.tran(edge[0]), enc_item.tran(edge[1]) , enc_feature.tran(edge[2]) , int(edge[3]) , int(edge[4])] for edge in hyper_edge]
    print("End Build Graph")
    interaction = [[int(edge[0]) , int(edge[1]) , int(edge[4])] for edge in unique_2dim_list(hyper_edge , [0,1])]
    print("End Build Graph")
        
    pos , neg = build_edge_from_hypergraph(args , hyper_edge)
    print("End Build Graph")
    hyper_edge = [ [edge[0] , edge[1]+nu , edge[2]+nu+ni , edge[3]] for edge in hyper_edge ]  

    graph["positive_edges"] = pos
    graph["negative_edges"] = neg
    graph["ecount"] = len(pos) + len(neg)
    graph["ncount"] = encoder['nu']+encoder['ni']+encoder['nf']
    graph["interaction"] = interaction # userid , itemid , rating  [seperate_id]
    graph["hyper_edge"] = hyper_edge   # userid , itemid , featureid , +/-1 [gather_id]

    print("End Build Graph")

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

def movie_len2formated_data(inpath , outpath) : 
    f = open(inpath , 'r')
    out = open(outpath , 'w')
    for line in f.readlines() : 
        sl = line.split('::')
        out.write(sl[0]+'\t'+sl[1]+'\t'+sl[2]+'\t'+'look|good|+1\t'+sl[3])
    f.close()
    out.close()


if __name__ == '__main__' : 
    XKLOG.testcase()
