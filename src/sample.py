import random
from utils import gather_2dim_list

class Sample (object) : 
    def __init__(self , num_user , num_item , neg2posratio) : 
        self.num_user , self.num_item = num_user , num_item
        self.neg2posratio = neg2posratio

    def sample(self) : 
        raise NotImplementedError('NotImplementedError')

class NegativeSample3(Sample) : 
    def sample(self , trainset) : 
        allitem = set(range(self.num_item))
        tmp = gather_2dim_list(trainset , 0)
        data = []
        for k , v in tmp.items() : 
            #import pdb
            #pdb.set_trace()
            pos_item = [item[1] for item in v]
            unknow_item = allitem - set(pos_item)
            for ipos in pos_item : 
                data.extend ([[k , ipos , random.choice(list(unknow_item))] for ttt in range(self.neg2posratio)])  # 可能有问题，但是问题不大

        print (len(data))
        return data
