#!/usr/bin/env python3
"""
    read through the output of a file in the output format of sxstats2.py
    and prepare a .CSV file, based on a table.
    First pass, input from stdin, output to stdout

    Each table line is a list entry.
    The first line is a list of input field numbers to group on
    following lines in the table describe a single field in the record/row/line
    descriptor lines are a two element list.  The first element is a list of
    leading field to match.  The second is the input field number to use as
    the content of the output field.

    The group of input lines consists of a successive lines of the
    input file with the same values in the "group on" fields.

    field numbers start with zero


    
"""
import sys

# sample table for width outputs
table = [ [0], # take a group of lines with the same first field
         [[None], 0], # None matches any field content. in fld 0 -> out fld 0
         [[None, 'out0'], 7], # if in field 1 == 'out0', in 7 -> out 1
         [[None, 'out1'], 7], # if in field 1 == 'out1', in 7 -> out 2
         [[None, 'out2'], 7], # if in field 1 == 'out2', in 7 -> out 3
         [[None, 'xf2'], 7], # if in field 1 == 'xf2', in 7-> out 4
         [[None, '*test*words*','N2'], 7], #three in f. to match
         [[None, 'n12'], 7]
        ]

def main():
    groupby = table[0]
    groupvals = [None]*len(groupby)
    group = [[]]
    outline = ['-']*(len(table)-1)
    outline_changed = False

    for lin in sys.stdin:
        inline = lin.strip().split()
        if matches(inline,groupby,groupvals):
            group.append(inline)
        else: # starting new group
            if outline_changed:
                print("\t".join(outline))
                outline_changed = False
                outline = ['-']*(len(table)-1)
            group = [inline]
            for i,f in enumerate(groupby):
                groupvals[i] = inline[i]

        # inline is added to current group; see if inline should generate
        # any output
        for i,e in enumerate(table):
            if i == 0: continue
            output = True
            for j,m in enumerate(e[0]):
                if m is None: continue # always a match
                if m == inline[j]: continue
                else:
                    output = False
                    break
            if output:
                outline[i-1] = inline[e[1]]
                outline_changed = True

    # possibly output last line of file
    if outline_changed:
        print("\t".join(outline))


def matches(inline,groupby,groupvals):
    """
        if the fields in inline indicated in groupby match the items in groupval
        then return true;  else false
    """
    for i,m in enumerate(groupby):
        if inline[m] == groupvals[i]:
            continue
        else:
            return False
    return True

main()
