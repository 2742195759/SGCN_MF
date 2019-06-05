import unittest 


class space (object) : 
    """simple test the namespace of the python
    """

    def __init__ (self) : 
        self.namespace = {}

    def __setitem__(self , key , value) : 
        self.namespace[key] = value

    def __getitem__(self , key) : 
        return self.namespace[key]


def plus(a , b) : 
    assert(isinstance(a , int) and isinstance(b , int))
    return a + b


class SimpleTest(unittest.TestCase) :
    '''simple test case of the unittest, use to increase the confident to my code. 
    '''

    def setUp(self): 
        print ('set up successfully')

    def tearDown(self): 
        print ('tear down successfully')
    
    def test_a_plus_b(self): 
        self.assertEqual(plus(1, 2), 3)
        with self.assertRaises(AssertionError) : 
            plus('str', 'kkk')
            pass

    def test_a_plus_b_large(self):
        self.assertEqual(plus(100000, 200000), 300000)


import torch 
import 

if __name__ == '__main__':
    #unittest.main()
    
