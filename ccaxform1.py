#!/usr/bin/env python3
"""
    reimplement in python the code from the CrossLingualSemantics group
    for canonical correlation transformation

    The original version of this code was in CCA.java and
    clss.transform.CanonicalCOrrelationTransformation.java

    That code mean-centers and unit-normalizes the embeddings before computing
    the transformation.   I don't know whether this causes problems,
    but based on our results, perhaps not.
    To compare results, I also did that.  

    I believe that Hardoon 
    and Artexte 
    and Press are relevant references for this version of the algorithm
"""
from gensim.models import KeyedVectors
import math
import numpy as np
import numpy.linalg as nl
import sys

def transforms(x, y): #x and y are matrices made up of corresponding dict entries
    """
        x and y are two vectors of vectors with x[n] being an embedding vector
        The width of x and y can vary, and don't have to match.
        x[n] is the embedding vector for a word w_n in L_x, and 
        y[n] is the embedding vector for a word v_n in L_y, such that 
           w_n and v_n are corresponding words

        The goal of CCA is to return two linear transforms A and B, 
          such that
            x @ A == y @ B,
          and this transform returns
            R = A @ (inverse B),
        which transforms a vector in the x space to one in the y space
            L_x[w_n]@ R  = x[n] @ R = y[n] = L_y[v_n]
        The point of returning R is the hope that we can find approximate
        L_y entries for L_x entries which do not appear in x
                 
   The java version of this code uses a compact SVD, 
   whereas numpy.linalg.svd is a full svd, 
   this matters for 'tall matrices' like x and y,

   In this version, I truncate the output of nl.svd, but probably I should 
   use a (hopefully faster) compact or truncating algorithm.

    """
    # get dimensions
    nx,mx = x.shape
    ny,my = y.shape
    typex = x.dtype

    if nx != ny or mx != my: squawk()

    # factorize x, y with svd.  x = ux @ np.diag(s) @ vhx
    ux, sx, vhx = nl.svd(x)
    uy, sy, vhy = nl.svd(y)

    #truncate ux, uy to n X m matrices
    ux = ux[:,:mx]
    uy = uy[:,:my]

    # turn the diagonal vectors into m by m matrices
    sx = np.diag(sx)
    sy = np.diag(sy)

    U1U2 = ux.T @ uy    # exciting variable name appears in java version of code

    up, sp, vhp = nl.svd(U1U2)

    # get inverses of sx and sy
    sxinv = nl.pinv(sx) 
    syinv = nl.pinv(sy)
    
    #pad inverses for multiply
    #Sxinv = np.zeros((x.shape[1],x.shape[0]) , dtype=np.float32)
    #Sxinv[:x.shape[1],:x.shape[1]] = sxinv
    #Syinv = np.zeros((y.shape[1],y.shape[0]) , dtype=np.float32)
    #Syinv[:y.shape[1],:y.shape[1]] = syinv

    # these are the matrices A and B mentioned in the top comment
    #    x @ A = y @ B
    A = vhx.T @ sxinv @ up
    B = vhy.T @ syinv @ vhp.T

    return A,B

def transform(X,Y):
    """
        call the Canonical Correlation Code above for the two transforms A,B`
        Then use  A and inverse of B to transform into the target space
    """
    A,B = transforms(X,Y)

    # compute the transform we promised.
    #  x @ A = y @ B

    binv = nl.pinv(B)
    R = A @ binv

    return R

def build_dict(filename, mx, my):
    """
        filename is the name of a translation dictionary,
          which has two words per line,
            first has vector in Keyedvectors mx
            second has vector in Keyedvectors my
        reads through the file,
        taking a pair of words from each line.
        builds two arrays, X and Y, to be used in transform function
    """
    lx = []
    ly = []
    # I specialized this version for the DSC task; identical words
    if type(filename) == type({}):
        for x in mx.keys():
            if not x in my: continue
            if x in filename: continue
            lx.append(mx[x])
            ly.append(my[x])
    
    else:
      with open(filename) as fi:
        for lin in fi:
            line = lin.strip().split()
            if len(line) != 2 : continue
            x,y = line
            if not x in mx: continue
            if not y in my: continue
            lx.append(mx[x])
            ly.append(my[y])

    X = np.ndarray((len(lx),len(lx[0])), dtype = np.float32)
    Y = np.ndarray((len(ly),len(ly[0])), dtype = np.float32)
    for i,(x,y) in enumerate(zip(lx,ly)):
        X[i] = x
        Y[i] = y
    return X,Y

def cu_normalize(kv):
    """
    kv is a gensim KeyedVector object.  It supports a unit_normalization,
    but not a centering one, hence this routine
    normalization is a side-effect, so nothing is returned.
    I trade off  space for a little extra computation time with the for loop
    I could skip doing that calculation and call kv.init_sim()
    which creates an array of norms first:
        dist = sqrt((m**2)sum(-1))[...,np.newaxis]
        m = m/dist
    but that makes two passes over all the vectors, which probably don't
    fit in cache.  I already made two while mean-centering
    """
    origin = kv.vectors.sum(0) / kv.vectors.shape[0]
    kv.vectors = kv.vectors - origin
    """\
    for i,v in enumerate(kv.vectors):
        x = math.sqrt(v.dot(v))
        kv.vectors[i] = v/x
    """ # this code significantly slower than following borrowed from gensim
    dist = sqrt((kv.vectors ** 2).sum(-1))[..., np.newaxis]
    kv.vectors /= dist

def main1():
    """
    test timing on normalization
    """
    import timeit
    from gensim.models import KeyedVectors
    import math

    setups = """\
import math
import numpy as np
from numpy import newaxis
from gensim.models import KeyedVectors
emb = KeyedVectors.load_word2vec_format('scratch/fsw1', binary=False)
"""

    print('starting timing0',flush=True)
    t = timeit.timeit(setup=setups,
                  stmt ='dist = np.sqrt((emb.vectors ** 2).sum(-1))[..., newaxis];'+
                        'emb.vectors /= dist', number = 1000)
    code1 = """\
for i,v in enumerate(emb.vectors):
    x = math.sqrt(v.dot(v))
    emb.vectors[i] = v/x
"""
    code2 = """\
m = emb.vectors
dist = np.sqrt((m ** 2).sum(-1))[..., newaxis]
m /= dist
"""
    
    print(t)  # this result about 7.8 on my machine
    print('starting timing1',flush=True)
    t=timeit.timeit(setup=setups,
        stmt = code1, number =1000)
    
    print(t) # this result about 98 on my machine .  So loop loses.
    print('starting timing2',flush=True)
    t=timeit.timeit(setup=setups,
        stmt = code2, number =1000)

    print(t) # about 7.6 on my machine
    print('finished timing',flush=True)

def main():
    """
    Test the transform code.   To make sure I am matching the java outputs,
    I am mean-centering and L2-normalizing, but since the original
    embeddings do not contain these normalizations it seems possible that 
    the resulting transformation may be different than if I hadn't done them.
    At least the s values in the svd should be different...
    """
    n = 2
    fn = 'trans.dict'
    fx = 'fsw1' #None
    fy = 'fsw2' #None
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    if len(sys.argv) > 4:
        fx = sys.argv[3]
        fy = sys.argv[4]

    # want to center and unit normalize the vectors
    mx = KeyedVectors.load_word2vec_format(fx,binary=False)
    my = KeyedVectors.load_word2vec_format(fy,binary=False)
    print("vectors loaded",file=sys.stderr,flush=True)
    
    cu_normalize(mx)
    cu_normalize(my)
    
    X,Y = build_dict(fn,mx,my)
    #Xs = X[:5001,:]   #this is a bug in the .java code, but i'll fix it
    #Ys = Y[:5001,:]   #only after I get the results to match
    print("starting build",file=sys.stderr,flush=True)
    R = transform(X,Y)
    for i in range(n):
        for j in range(n):
            print(R[i,j],end=' ')
        print()

if __name__ == '__main__': main()
