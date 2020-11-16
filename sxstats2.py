#!/usr/bin/env python3
"""
    13/11/20
    add an optional decimal number as first argument on command line.  (means no
    all-digit folder paths)
    bits of the number are binary options.
        (mask of 1)  hide target summaries
        (mask of 2)  display testword ranks, etc.
        (mask of 4)  display summaries by width

    read through a list of directories specified on the command line, and
    for each find the nested xform_stats.txt files.
    summarize the statistics in the files by width:
        noting, 
            for the 100 test words,  and 
            for the twelve "unchanged" target words, and
            for the six "changed" target words:

                average rank
                average distance
                average of the three "neighborly" statistics
                    neighbormatch(t0, t1)
                    neighbormatch(t0, t1*xf)
                    neighbormatch(t1, t1*xf)

        for the test words, most of the ranks are zero. do we get interesting
        differences if we segregate by rank?
"""
import stats
import sys

class statsrec:
    def __init__(self):
        self.rank = stats.stats()
        self.distance = stats.stats()
        self.n0,self.n1,self.n2 = stats.stats(),stats.stats(),stats.stats()

    def setdistance(self,r, id=''):
        self.distance. newitem(r, id)
        self.distancev= r

    def setrank(self,r, id=''):
        self.rank. newitem(r, id)
        self.rankv= r

    def setnei(self,r, id=''):
        self.neiv= r
        self.n0. newitem(r[0], id)
        self.n1. newitem(r[1], id)
        self.n2. newitem(r[2], id)

    def setouts(self,v,ide = ''):
        """
            here and following I assume object is implemented with a dictionary
            This low-level fooling feels quite unpythonic, but
            the description of hasattr() in pydoc3 includes implementation
            detail:  try getattr and catch the attribute error
            and I feel more comfortable playing with the dict.
            
        """
        for i,k in enumerate(['out0','out1','out2']):
            if not k in self.__dict__:
                self.__dict__[k] = stats.stats(k)
            st = self.__dict__[k]
            st.newitem(v[i],ide)

    def setxf2(self,v,ide):
        if not hasattr(self,'xf2'):     #not defined(self.xf2):
            self.xf2 = stats.stats('xf2')
        self.xf2.newitem(v, ide)

    def setf(self,f,v):
        """
        set a field, f, to a value, v
        """
        self.__dict__[f] = v

class defdict (dict) :  # if I got the syntax right, inherit from dict...
    """
    a class providing the interface of a dictionary, but with the property
    that when you create the class, you provide a constructor for the
    default content, 
    and a reference to a blank cell will always return an item
    freshly built with that constructor.
    """

    def __init__(self,arg_constructor):
        self.arg_constructor = arg_constructor

    def __getitem__(self,k):
        val = self.get(k, None)    # None value will do for blank?
        if type(val) == type(None): # seems like Nonetype should be constant
            val = self.arg_constructor()#k)
            self[k] = val
        return val

def main():
    """
    read and summarize xformstat.txt files from folders on command-line
    following (optional) numeric option 
    """
    # 
    if len(sys.argv) > 1 and all([c>='0' and c<='9' for c in sys.argv[1]]):
        options = int(sys.argv[1])
        dirs = sys.argv[2:]
    else:
        options = 0
        dirs = sys.argv[1:]

    widict = defdict(statsrec)
    widwordict = defdict(statsrec) #dict()
    wordict = defdict(statsrec) #dict()
    goldict, goldlist = readict('gold1.txt')

    for di in dirs:
        wbeg = di.find('/')+1
        widend = di.find('-')
        wid = di[wbeg:widend]
        itend = di.find('-',widend+1)
        jobend = di.find('+',itend+1)
        job = di[itend+1:jobend]
        wjob = wid+'|'+job
        widrec = widict.get(wid,0)
        if widrec == 0:
            widict[wid] = widred = statsrec()
        with open(di+'/xform_stats.txt') as fi:
            tmpwordict = defdict(statsrec) #dict()
            for lin in fi:
                if lin == '\n':continue
                if startswith('transdict_total:', lin): continue
                if startswith('TestWords:', lin): continue 
                if startswith('Neighborliness:',lin):
                    #print(wid, lin)
                    neighs = valfrom(15,lin)
                    widict[wid].setnei(neighs,wjob)
                    continue
                if startswith('Outputs=',lin):
                    #print(wid, lin)
                    outputs = valfrom(8,lin)
                    widict[wid].setouts( outputs,wjob)
                    continue
                if startswith('2-norm xform=',lin):
                    #print(wid, lin)
                    xformnorm = valfrom(13,lin)[0]
                    widict[wid].setxf2( xformnorm,wjob)
                    continue
                if startswith('r(',lin):
                    word, target, values = gword(lin)
                    rank = values[0]
                    widwordict[(wid,word)].setrank(rank)
                    wordict[word].setrank(rank)
                    if not target:
                        widwordict[(wid,word)].setrank(rank,wjob)
                        tmpwordict[(wid,word)].setrank(rank,wjob)
                    continue
                if startswith('t(',lin):
                    w1, target, (t01,t02,t12) = gword(lin)
                    if w1 != word:
                        gripe()
                    neighsw=[t01,t02,t12]
                    widwordict[(wid,word)].setnei(neighsw)
                    wordict[word].setnei(neighsw)
                    if not target:
                        widwordict[(wid,word)].setnei(neighsw)
                        tmpwordict[(wid,word)].setrank(rank,wjob)
                    continue
                if startswith('d(',lin):
                    w1, target, (distance,) = gword(lin)
                    if w1 != word:
                        gripe()
                    widwordict[(wid,word)].setdistance(distance)
                    wordict[word].setdistance(distance)
                    if not target:
                        widwordict[(wid,word)].setdistance(distance)
                        tmpwordict[(wid,word)].setdistance(distance)
                # print out single CSV line for word
                #if target:
                #    gold = goldict[word]
                #else:
                #    gold = None
                #print(wid, word, target, gold, rank, distance, t01,t02,t12)
        #here finished single job, (which has single width).  
        # could print job summary using tmpwordict, wid
    #here finished with all di, that is all requested files.  Print summary
    if test(4,options): # display per width summaries
        # width globals
        testwords = defdict(list)
        for wid,word in widwordict.keys():
            testwords[wid].append(word)
        widths = list(widict.keys())
        widths.sort()
        for w in widths:
            rec = widict[w]
            stats_w(w,rec)
            # show the width-only stats for output and xform
            for i,k in enumerate(['out0','out1','out2', 'xf2']):
                if hasattr(rec,k):
                    print(w,k,end=' ')
                    st = getattr(rec,k)
                    st.print()
            # build composite word statistics for this width
            comp = statsrec()
            for wd in testwords[w]:
                if wd in goldict: continue # ignore targets!
                stw = widwordict[(w,wd)]
                for crec,rec in [(comp.rank,stw.rank),(comp.distance,stw.distance),(comp.n0,stw.n0),(comp.n1,stw.n1),(comp.n2,stw.n2)]:
                    crec.mergestats(rec)
            for crec,desc in [(comp.rank,'Rank '),(comp.distance,'Dist '),(comp.n0,'N0 '),(comp.n1,'N1 '),(comp.n2,'N2 ')]:
                print(w,'*test*words*',desc,end = ' ')
                crec.print()

    if test(2,options):  # print test words, but not targets
        for w,rec in wordict.items():
            if w in goldict: continue 

    if not test(1,options): # print targets?
     for change in '0','1':
      for w in goldlist:
       if goldict[w] == change:
        print(w+' Chng',change)
        rec = wordict[w]
        stats_w(w,rec)          # show the stats


def stats_w(w,rec):
    """
        print out statistics for a single word, either a testword or a target
    """
    for desc,stat in [('Rank',rec.rank),('Dist',rec.distance),('n01',rec.n0),('n02',rec.n1),('n12',rec.n2)]:
        if stat.minval is not None:     #if no newitem ever added
            print(w+' '+desc,end = ' ')
            stat.print()


def startswith(string, line):
    """
    return True or False, depending on whether line begins with string
    """
    lens = len(string)
    if line[:lens] == string: return True
    return False


def gword(string):
    """
        get word, whether it is in the target list, and some sample values
        from a string in the form:
        x(<word>)* val val val\n
    """
    wend = string.find(')')
    word = string[2:wend]
    if string[wend+1] == '*':
        target = True
        wend += 1
    else: target = False
    values = valfrom(wend+1,string)

    return word,target,values
    
def valfrom(fro, string):
    """
      return a list of float values from string 
      which begin at character position fro
    """
    values = string[fro:].strip().split()
    for i,v in enumerate(values):
        if v == 'None':
            values[i] = 0
        else:
            values[i] = float(v)
    return values

def readict(fn):
    """
        read in a dictionary file, where each line has a key and a value
        return dictionary, and list of keys in original order
    """

    dic = dict()
    lis = []
    with open(fn) as fi:
        for lin in fi:
            line = lin.strip().split()
            if len(line) != 2:
                complain()
            dic[line[0]] = line[1]
            lis.append(line[0])

    return dic,lis

def statsum(desc,st):
    """
        print summary of the stats object with descriptor desc
    """
    print (desc, end=' ')
    st.print()

def test(mask, options):
    """
        return true if mask is on in options
    """
    if (options & mask) == mask: return True
    return False



if __name__ == '__main__': main()
