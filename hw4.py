import numpy as np
import itertools
import math

cs = ['e', "j", "s"]
chars = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"," "]
training_data = []
training_labels = []
for lang in cs:
    for i in range(10):
        with open(f"languageID/{lang}{i}.txt") as f:
            curr_data = [line.rstrip() for line in f]
            curr_data = list("".join(curr_data))
            curr_data = [x for x in curr_data if x in chars]
            training_data += [curr_data]
            training_labels += [lang]
            
smooth = 0.5
def estimate_prior(lang):
    return ((np.array(training_labels) == lang).sum() + smooth)/(len(training_labels) + len(cs) * smooth)

e_prior = estimate_prior("e")
j_prior = estimate_prior("j")
s_prior = estimate_prior("s")

def estimate_conditional(label):
    estimates = {}
    inds = np.where(np.array(training_labels) == label)
    class_data = np.array(list(itertools.chain.from_iterable(list(np.array(training_data)[inds]))))
    for char in chars:
        curr = ((class_data == char).sum() + smooth) / (len(class_data.flatten()) + len(chars) * smooth)
        print(f"p(c_i={char}|y={label}) = {curr} \\\\")
        estimates[char] = curr
    return estimates

english_conditionals = estimate_conditional("e")
japanese_conditionals = estimate_conditional("j")
spanish_conditionals = estimate_conditional("s")

def bow_count(fn, print=False):
    with open(fn) as f:
        curr_data = [line.rstrip() for line in f]
        curr_data = list("".join(curr_data))
        curr_data = [x for x in curr_data if x in chars]
    counts = []
    count_dict = {}
    for char in chars:
        curr_total = (np.array(curr_data) == char).sum()
        counts += [curr_total]
        count_dict[char] = curr_total
    if print:
        print(count_dict)
    return count_dict

char_counts = bow_count(f"languageID/e10.txt", print=True)

def document_conditional(counts, conditionals):
    total = 0
    for char in chars:
        total += counts[char] * math.log(conditionals[char])
    return total

xe10_cond_e = document_conditional(char_counts, english_conditionals)
xe10_cond_j = document_conditional(char_counts, japanese_conditionals)
xe10_cond_s = document_conditional(char_counts, spanish_conditionals)

p_e = xe10_cond_e + math.log(e_prior)
p_j = xe10_cond_j + math.log(j_prior)
p_s = xe10_cond_s + math.log(s_prior)


def make_pred(char_counts):
    xe10_cond_e = document_conditional(char_counts, english_conditionals)
    xe10_cond_j = document_conditional(char_counts, japanese_conditionals)
    xe10_cond_s = document_conditional(char_counts, spanish_conditionals)
    p_e = xe10_cond_e + math.log(e_prior)
    p_j = xe10_cond_j + math.log(j_prior)
    p_s = xe10_cond_s + math.log(s_prior)
    best = max([p_e, p_j, p_s])
    if best == p_e: return "e"
    elif best == p_j: return "j"
    else: return "s"
    
#Confusion matrix
conf = {}
for pred in cs:
    for target in cs:
        conf[f"{pred}{target}"] = 0
        

for lang in cs:
    for i in range(10, 20):
        fn = f"languageID/{lang}{i}.txt"
        with open(fn) as f:
            curr_data = [line.rstrip() for line in f]
            curr_data = list("".join(curr_data))
            curr_data = [x for x in curr_data if x in chars]
        char_counts = bow_count(fn)
        pred = make_pred(char_counts)
        conf[f"{pred}{lang}"] += 1
        
