'''
Script to read in the data sets
Wren Saylor 2019
'''

import pandas as pd
from Bio import SeqIO,Seq
import numpy as np
import random

def reverse_complement_dictionary(sequence):
    '''
    dicationary with reverse complement nucs
    '''
    seqDict = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
    return "".join([seqDict[base] for base in reversed(sequence)])

def read_from_csv(file,value):
    '''
    for those data sets from csv files, read into panda
    add sequence and probability columns, get revese complement to boost #s
    '''
    pd_seq = pd.read_csv(file,header=None)

    # label columns and make probabilty
    pd_seq.columns = ['sequence']
    pd_seq['probability'] = value

    # make binary sequence, where each letter is coded as 4 digit 0/1
    pd_seq['binary'] = [make_binary_sequence(seq) for seq in pd_seq['sequence']]
    return pd_seq

def read_from_fa(file):
    '''
    for those data sets from fa files (the negative sequences)
    '''
    records = SeqIO.parse(file, "fasta")

    # get just the sequences from the record
    collect = []

    # collect just the sequences
    for r in records:
        seq = ''.join(r.seq)
        collect.append(seq)
    return pd.DataFrame({'sequence':collect})

def remove_positives(pos,neg):
    '''
    remove all negative sequences that contain positives
    make panda for which positive and negative sequences overlap,
    incase need later
    '''
    collect = []

    # for each row in pos sequences, go through the negative rows
    # if the positive sequece is in the negative, collect those indeces to
    # overlap list, other wise, collect the sequence
    pos_seq = pos['sequence'].tolist()
    neg_seq = neg['sequence'].tolist()
    for n in neg_seq:
        for p in pos_seq:
            collect.append(n.replace(p, ""))
    pd_seq = pd.DataFrame({'sequence':collect})
    return pd_seq

def subsample_negative(pos,neg,size):
    '''
    get the same number of negative sequence as have posistive
    with same size elements (17)
    make sure the sequences are not already in the positive set
    make into panda with columns sequence and probablity
    '''
    collect = []
    k = 17 # length of binding site
    pos_list = pos['sequence'].tolist()

    # for each negative row, while the number of sequences in the collected list
    # is less than the size indicated to collect, continue collecting,
    # unless the negative sequence is in the positive list
    for i,n in neg.iterrows():
        while len(collect) < size:
            neg_str = n['sequence']
            start = random.randint(0,len(neg_str)-k)
            sub_string = neg_str[start:start+k]
            if sub_string in pos_list:
                continue
            else:
                collect.append(sub_string)
    pd_seq = pd.DataFrame(collect)
    pd_seq.columns = ['sequence']
    pd_seq['probability'] = 0.0
    return pd_seq

def select_negatives(neg_file,seq_pos):
    '''
    run all the functions necessary to create the negative sequences data set
    formated in the same way and complementry to the positives
    '''
    neg_read_fa = read_from_fa(neg_file)
    neg_no_overlaps = remove_positives(seq_pos,neg_read_fa)
    neg_subsample = subsample_negative(seq_pos,neg_no_overlaps,1000)#len(pos_list*2)
    neg_subsample['binary'] = [make_binary_sequence(seq) for seq in neg_subsample['sequence']]
    return neg_subsample

def make_binary_sequence(seq):
    '''
    convert the sequence into binary by exchanging each letter with a four
    digit 1/0 code
    '''
    bin_seq = []
    bin_dict = {"A": [1,0,0,0],"C": [0,1,0,0],"G": [0,0,1,0],"T": [0,0,0,1]}
    temp = [bin_seq.append(bin_dict[s]) for s in seq]
    return np.array(bin_seq).flatten()

def make_training_and_holdout(df):
    '''
    shuffle the dataframe, then seperate into training and test sets with ratio 7:3
    '''
    # shuffle the entire df
    df_shuffle = df.sample(frac=1).reset_index(drop=True)

    # the cutoff value for seperating into training and test
    # is at 70% of the seqneces
    cutoff = int(len(df_shuffle)*.7)
    train = df_shuffle[:cutoff]
    test = df_shuffle[cutoff:]
    return train,test

def assemble_positive_dataset(file_pos):
    '''
    get the positive sequendes and their reverse complements to extend the set
    '''
    # get positive sequences
    read_pos = read_from_csv(file_pos,1.0)

    # revrse complement
    reverse = [reverse_complement_dictionary(i) for i in read_pos['sequence']]
    pd_rev = pd.DataFrame(reverse,columns=['sequence'])
    pd_rev['probability'] = 1.0
    pd_rev['binary'] = [make_binary_sequence(seq) for seq in pd_rev['sequence']]

    # concat with reverse complement
    seq_pos = pd.concat([read_pos,pd_rev])
    return seq_pos

def collect_datasets(file_pos,file_neg,file_test):
    '''
    get positive, negative, and test datasets
    all in pandas, with sequence and probability columns
    '''

    # get test sequences
    seq_test = read_from_csv(file_test,np.nan)

    # get positive sequences
    seq_pos = assemble_positive_dataset(file_pos)

    # get negative sequences
    seq_neg = select_negatives(file_neg,seq_pos)

    # subsample the positive sequences in order to get the same number of pos
    # as there are neg
    collect = []
    num_pos = len(seq_neg)-len(seq_pos)
    for i in range(num_pos):
        collect.append(seq_2.sample(1))
    sample_pos = pd.concat(collect)

    # concat pos and neg seq together
    seq_all = pd.concat([seq_pos,seq_neg,sample_pos])

    # get training and test sequences
    seq_train,seq_hold = make_training_and_holdout(seq_all)

    return seq_train,seq_hold,seq_test
