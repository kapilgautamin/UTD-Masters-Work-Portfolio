#Authored by: Kapil Gautam - KXG180032

def add_to_dict(prev,new):
    for item in new:
        if item in prev:
            prev[item] += 1
        else:
            prev[item] = 1
    return prev

def make_bigrams(sentence):
    bigram_from_sentence = {}
    token = sentence.split()
    for pos in range(len(token)-1):
        if token[pos] not in seperators and token[pos] not in weird:
            if token[pos+1] not in seperators and  token[pos+1] not in weird:
                if (token[pos],token[pos+1]) not in bigram_from_sentence:
                    bigram_from_sentence[token[pos],token[pos+1]] = 1
                else:
                    bigram_from_sentence[token[pos],token[pos+1]] += 1
    return bigram_from_sentence

def add_to_vocab(sentence):
    vocab_from_sentence = {}
    token = sentence.split()
    for pos in range(len(token)):
        if token[pos] not in seperators and token[pos] not in weird:
            if token[pos] not in vocab_from_sentence:
                vocab_from_sentence[token[pos]] = 1
            else:
                vocab_from_sentence[token[pos]] += 1
    return vocab_from_sentence


def bigram_probability(a,b,one_smoothing):
    #p(a|b)
    #p("want"|"i") = "i want" = C(bigram("i","want")) / C(vocab("i"))
    if one_smoothing:
        # print("One smoothing is: ", one_smoothing)
        if (a,b) in bigrams:
            return (bigrams[a,b] + 1) / (vocab[a] + len(vocab))
        else:
            return 1 / (vocab[a] + len(vocab))
    else:
        if (a,b) in bigrams:
            return bigrams[a,b] / vocab[a]
        else:
            return 0 / vocab[a]

def make_matrix(sentence,one_smoothing):
    token = add_to_vocab(sentence)
    matrix_bigrams = make_bigrams(sentence)
    print("Printing the bigram counts for each word of sentence: \"" + sentence + "\"")
    print("%12s"%(""),end="")
    for header in token:
        print("%12s"%(header),end="")
    print()
    for y in token:
        print("%12s"%(y), end="")
        for i in token:
            # print (y,i, end="")
            if one_smoothing:
                if (y,i) in bigrams:
                    print("%12d"%(bigrams[y,i]+1), end="")
                else:
                    print("%12d"%(1), end="")
            else:
                if (y,i) in bigrams:
                    print("%12d"%(bigrams[y,i]), end="")
                else:
                    print("%12d"%(0), end="")
        print()
    print()

def make_prob_matrix(sentence,one_smoothing):
    token = add_to_vocab(sentence)
    matrix_bigrams = make_bigrams(sentence)
    print("Printing the bigram probabilities for each word of sentence: \"" + sentence + "\"")
    print("%12s"%(""),end="")
    for header in token:
        print("%12s"%(header),end="")
    print()
    for y in token:
        print("%12s"%(y), end="")
        for i in token:
            # print (y,i, end="")
            print("%12f"%(bigram_probability(y,i,one_smoothing)), end="")
        print()
    print()


def sentence_probability(sentence,one_smoothing):
    import math
    sen_bigrams = make_bigrams(sentence)
    final_prob = 1
    log_prob = 0
    make_matrix(sentence,one_smoothing)
    make_prob_matrix(sentence,one_smoothing)
    for bi in sen_bigrams:
        # print(bi, bigram_probability(bi[0],bi[1],one_smoothing))
        final_prob *= bigram_probability(bi[0],bi[1],one_smoothing)
        log_prob += math.log(final_prob) * -1
    print("The probability of sentence \"" + sentence + "\" is: " + str(final_prob))
    print("The Log probability of sentence \"" + sentence + "\" is: " + str(log_prob))
    return final_prob

if __name__ == '__main__':
    import sys
    #default variables
    training_file = "train.txt"
    testing_file = "test.txt"
    one_smoothing = True 

    #rewrite input from command line
    if len(sys.argv) == 4:
        training_file = sys.argv[1]
        testing_file = sys.argv[2]
        one_smoothing = int(sys.argv[3])
    
    #The following debugging options can be turned on when debugging depending on the verbosity required.
    DEBUG = False
    DEBUG_VERBOSE = False
    DEBUG_SUPER_VERBOSE = False
    train = open(training_file,"r")

    bigrams = {}
    vocab = {}
    #Taking : . , ; ?as seperate words and not as seperators
    seperators = "\"[]-'()! "
    weird = ['",',':"', '"--', '"...', '...', '":', '";', '"`', '"`...', "',", "'-", '),', ',"', ",'", ',\'"', ',--',
    '--', '--"', '--\'"', '--,', '--`', '.,', '.--', '..."', "...'", '....', "....'",'...."', '1914', '379',
    '".','!"', ':,', '`','?\'"','."',"'.",".'",'.\'"','?"',"--'","!'",'--?"',"?'"]
    if DEBUG_VERBOSE:
        count_lines = 0
    #Each line is given as a sentence
    for line in train:
        #We cannot use the update method as it will add two dictionaries, but it will overwrite the previous keys
        # bigrams.update(make_bigrams(line))
        # vocab.update(add_to_vocab(line))

        #augment start and stop symbols with spaces
        line = "<s> " + line + " </s>"
        bigrams = add_to_dict(bigrams, make_bigrams(line))
        vocab = add_to_dict(vocab, add_to_vocab(line))

        if DEBUG_VERBOSE:
            count_lines += 1
            if count_lines == 100:
                # print (bigrams)
                print(vocab)
                break;

    if DEBUG_SUPER_VERBOSE:
            print(vocab)
            print("\n\n\n\n")
            print(bigrams)

    if DEBUG_VERBOSE:
        print(sorted(vocab))
        print("\n\n\n\n")
        print(sorted(bigrams))

    if DEBUG:
        print("bigrams['i','want'] = " + str(bigrams['i','want']))
        print("vocab['i'] = " + str(vocab['i']))
        print("bigram_probability('want','i') = " + str(bigram_probability("i","want",True)))

        print("bigrams['a','black'] = " + str(bigrams['a','black']))
        print("vocab['a'] = " + str(vocab['a']))
        print("bigram_probability('black','a') = " + str(bigram_probability("a","black",True)))

    # sentence = "i do not think this young lady is so celtic as i had supposed"
    # make_matrix(sentence,one_smoothing)

    test = open(testing_file,"r")
    for line in test:
        #augment start and stop symbols with spaces
        line = "<s> " + line + " </s>"
        sentence_probability(line,one_smoothing)
        print("---------------------------------------------------------------------------------------------")