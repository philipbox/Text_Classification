import operator
import os
import re
import math
CONDITION = ["NNG", "NNP"]

global_word_set = set()
global_idf_dic = {}
top_5000_words = []

class Document(object):
    def __init__(self):
        self.file_path = ""
        self.category = ""
        self.terms = {}
        self.tf = {}
        self.tf_idf = {}
        self.tf_idf_norm = {}

    def push_terms(self, terms_list):
        for term in terms_list:
            if term in self.terms:
                self.terms[term] += 1
            else:
                self.terms[term] = 1

    def cal_tf(self):
        for term_key in self.terms:
            self.tf[term_key] = math.log10(self.terms[term_key]+1)

    def cal_tf_idf(self):
        sum = 0
        for word in top_5000_words:
            term = word[0]
            try:
                self.tf_idf[term] = self.tf[term] * global_idf_dic[term]
                sum += (self.tf_idf[term]**2)
            except KeyError as key:
                self.tf_idf[term] = 0.0

        for word in top_5000_words:
            term = word[0]
            try:
                self.tf_idf_norm[term] = self.tf_idf[term]/math.sqrt(sum)

            except ZeroDivisionError:
                self.tf_idf_norm[term] = 0.0

def print_document(document_list):
    for document in document_list:
        print(document.category, document.file_path)
        print(*document.terms.items())
        input()

def file_read(dir):
    document_list = []
    for root, dir, fnames in os.walk(dir):
        for fname in fnames:
            full_file_path = root + "/" + fname;
            file = open(full_file_path, "r", encoding="utf-8")
            document = Document()
            document.file_path = full_file_path
            document.category = root.split('\\')[-1]

            lines = file.readlines()
            for line in lines:
                terms = line.split("\t")
                if len(terms) <= 1:
                    continue
                terms = [re.sub('[^가-힝0-9a-zA-Z/]', '', str) for str in terms[1].split("+")]
                valid_terms = [t.split("/")[0] for t in terms if len(t.split("/")) >= 2 and (t.split("/")[1] in CONDITION)]
                valid_terms = [term for term in valid_terms if len(term) >= 2]
                global_word_set.update(valid_terms)
                document.push_terms(valid_terms)
            document_list.append(document)
    return document_list

def calculate_tf(document_list):
    for document in document_list:
        document.cal_tf()

def calculate_idf(document_list):
    total_document_num = len(document_list)
    for word in global_word_set:
        global_idf_dic[word] = math.log10(total_document_num/(len([document for document in document_list if word in document.terms ])+1))

def get_rank():
    sorted_idf = sorted(global_idf_dic.items(), key=operator.itemgetter(1, 0))
    top_5000_rank = sorted_idf[0:5000]

    for term in top_5000_rank[:30]:
        print(term)

    return top_5000_rank

def calculate_tf_idf(document_list):
    for document in document_list:
        document.cal_tf_idf()

def file_save(document, save_path):
    save_dir = save_path + "/" + document.file_path.split("\\")[-1]

    if not os.path.exists(save_path + "/"+ document.category):
        os.makedirs(save_path + "/"+ document.category)

    file = open(save_dir, mode="w")
    for words in top_5000_words:
        word = words[0]
        try:
            file.write(str(document.tf_idf_norm[word])+'\t')
        except KeyError:
            file.write( str(0) + "\t" )
    file.close()

def files_save(document_list, save_path):
    for document in document_list:
        file_save(document, save_path)

def TF_IDF_PROCESSING(dir, save_dir, isTraining=False):
    document_list = file_read(dir)
    calculate_tf(document_list)

    if isTraining:
        calculate_idf(document_list)
        top_5000_words.extend([i for i in get_rank()])

    calculate_tf_idf(document_list)

    files_save(document_list, save_dir)

if __name__ == "__main__":
    TF_IDF_PROCESSING("./Input_Data_input", "./RES_TrainingSet", True) # Training Data Set
    TF_IDF_PROCESSING("./Test_Data_input", "./RES_TestingSet", False) # Test Data set
