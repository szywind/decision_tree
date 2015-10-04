import numpy as np
import numpy.linalg as nplg
import os
import math
#import pp


with open("./hw3_train.dat","rt") as fl:
    dataTr = []
    for line in fl:
        tmp = [float(elem) for elem in line.strip('\n').split()]
        dataTr.append(tmp)  
    train = np.array(dataTr)

with open("./hw3_test.dat","rt") as fl:
    dataT = []
    for line in fl:
        tmp = [float(elem) for elem in line.strip('\n').split()]
        dataT.append(tmp)  
    test = np.array(dataT)

Ntr = train.shape[0]
Nt = test.shape[0]



def Gini_index_slow(y1,y2):
    N1 = y1.shape[0]
    N2 = y2.shape[0]
    return N1*(1 - (sum(y1 == 1)/N1)**2 - (sum(y1 == -1)/N1)**2) + N2*(1 - (sum(y2 == 1)/N2)**2 - (sum(y2 == -1)/N2)**2)    

def Gini_index(y1,y2):
    N1 = y1.shape[0]
    N2 = y2.shape[0]
    n1 = sum(y1 == 1)
    n2 = sum(y2 == 1)
    return n1*(N1-n1)/float(N1) + n2*(N2-n2)/float(N2)    

## Nodes = []
def DecisionTree(dataSet, prune = False):
    
    gt_left = []
    gt_right = []
    N = dataSet.shape[0]
    if 1:
        #print "train = ", dataSet
        #print "adf = ", dataSet[:,-1].max() == dataSet[:,-1].min() or sum(dataSet.min(0)-dataSet.max(0)) == 0  
        if dataSet[:,-1].max() == dataSet[:,-1].min() or sum(dataSet[:,:-1].min(0)-dataSet[:,:-1].max(0)) == 0:
            if sum(dataSet[:,-1] == -1) > sum(dataSet[:,-1] == 1):
                return [-1]
            else:
                return [1]
        else:

        # calculate gt
            best_val = N*1.0
            gt = [-1, -1, -1]
            theta = -1000
            # traverse all features
            for i in range(dataSet.shape[1]-1):
                dataSetSorted = dataSet[dataSet[:,i].argsort()]                
                # traverse all theta
                for j in range(N-1):

                    #theta = 0.5*(dataSetSorted[j,i] + dataSetSorted[j+1,i])
                    #Dc_left = dataSet[dataSet[:,i]<theta,]
                    #Dc_right = dataSet[dataSet[:,i]>theta,]
                    Dc_left = dataSetSorted[:j+1,]
                    Dc_right = dataSetSorted[j+1:,] 
                    
                    impurity = Gini_index(Dc_left[:,-1], Dc_right[:,-1])
                    if impurity < best_val:
                        assert(j != -1)
                        if j==-1:
                            theta = dataSetSorted[j+1,i] - 100
                        else:
                            theta = 0.5*(dataSetSorted[j,i] + dataSetSorted[j+1,i])
                        best_val = impurity
                        gt = [i,theta]
            ## Nodes.append(gt)
            if not prune:
                gt_left = DecisionTree(dataSet[dataSet[:,gt[0]]<gt[1],])
                gt_right = DecisionTree(dataSet[dataSet[:,gt[0]]>gt[1],])
                return [gt, [gt_left,gt_right]]
            else:

                left_label = int(np.sign(sum(dataSet[dataSet[:,gt[0]]<gt[1],-1])))
                right_label = int(np.sign(sum(dataSet[dataSet[:,gt[0]]>gt[1],-1])))
                return [gt,[[left_label],[right_label]]]
    
        
## -------------- Q13 -------------------------   
G = DecisionTree(train)
def computeNode(G):
    Nodes = [] 
    if len(G) == 2:
        if type(G[0]) == int and type(G[1]) == np.float64 or type(G[1]) == float:
            return [G]
        elif len(G[0]) == 1 and len(G[1]) == 2:
            Nodes.extend(computeNode(G[1]))
        elif len(G[0]) == 2 and len(G[1]) == 1:
            Nodes.extend(computeNode(G[0]))
        elif len(G[0]) == 2 and len(G[1]) == 2:
            Nodes.extend(computeNode(G[0]))
            Nodes.extend(computeNode(G[1]))

    return Nodes
    
# Q13
n_nodes = computeNode(G)
print "total # of internal nodes is: ", len(n_nodes)


## --------------- Q14-Q15 --------------------  
def traverseDecisionTree(x, gt, Nodes):
    if len(gt) == 2:
        if gt[0] in Nodes:
            if x[gt[0][0]] < gt[0][1]:
                y_pred = traverseDecisionTree(x,gt[1][0], Nodes)
            else:
                y_pred = traverseDecisionTree(x,gt[1][1], Nodes)
        else:
            assert (gt[1] in Nodes)
            if x[gt[1][0]] < gt[1][1]:
                y_pred = traverseDecisionTree(x,gt[0][0], Nodes)
            else:
                y_pred = traverseDecisionTree(x,gt[0][1], Nodes)
        return y_pred
        
    else:
        return gt[0]
        
            
            
def runDecisionTree(dataSet,G, Nodes):
    N = dataSet.shape[0]
    E = 0
    for i in range(N): 
        y_pred = traverseDecisionTree(dataSet[i,:-1], G, Nodes)
        if y_pred != dataSet[i,-1]:
            E += 1
            
    return E/(N+0.0)
        
# Q14
runDecisionTree(train, G, n_nodes)
# Q15
runDecisionTree(test, G, n_nodes)


## --------------- Q16-Q20 --------------------      
T = 300
R = 100
import random
def bagging(dataSet, DecisionTree, prune = False):
    Node_all = []
    gt_all = []
    N = dataSet.shape[0]
    for r in range(R):
        for t in range(T):
            sampleId = random.sample(range(N)*N, N)
            #for i in range(N):
            #    sampleId.append(random.randrange(N))
            dataNew = dataSet[sampleId,:]
            gt = DecisionTree(dataNew, prune)     
            gt_all.append(gt)
            Node_all.append(computeNode(gt))
    assert(len(gt_all) == T*R)
    assert(len(Node_all) == T*R)
    return (gt_all, Node_all)
 
            
def runBagging(dataSet, average = False, prune = False):
    gt_all, Node_all = bagging(train, DecisionTree, prune)

    N = dataSet.shape[0]
    mu_Error = 0
    for r in range(R):
        Error = 0
        for i in range(N):
            y_pred = np.zeros([T,1])
            for t in range(T):
                #print gt_all[r*T+r]
                #print ""
                #print Node_all[r*T+t]
                
                y_pred[t] = traverseDecisionTree(dataSet[i,:-1], gt_all[r*T+t], Node_all[r*T+t])
            if not average:
                if np.sign(sum(y_pred)) != dataSet[i,-1]:
                    Error += 1
            else:
                tmp = sum(y_pred != dataSet[i,-1])
                Error += tmp/(T+0.0)
        mu_Error += Error
    mu_Error /= (R+0.0)
    return mu_Error/(N+0.0)   

# Q16
print "Q16 ", runBagging(train, average = True) 
# Q17    
print "Q17 ", runBagging(train)
# Q18
print "Q18 ", runBagging(test)
# Q19
print "Q19 ", runBagging(train, prune = True)
# Q20
print "Q20 ", runBagging(test, prune = True)


                