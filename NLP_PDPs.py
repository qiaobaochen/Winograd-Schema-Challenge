import xml.etree.ElementTree as etree
import spacy
import numpy as np
from conceptNet import conceptNet


'''
import data and put it in problems list
'''
def wsc_pdps_input_data():
    # import data from WSCollection.xml 
    tree = etree.parse('data/WSCollection.xml')
    root = tree.getroot()
    problems = list()
    original_problems = root.getchildren()
    #parse original_problems to problem and append it to problems list 
    for original_problem in original_problems:
        problem = dict()
        for information in original_problem.getchildren():
            if information.tag == 'answers':
                answers = information.getchildren()
                answer_list = list()
                for answer in answers:
                    answer_list.append(answer.text.strip())
                problem['answers'] = answer_list
            elif information.tag == 'text':
                texts = information.getchildren()
                text_dict = dict()
                for text1 in texts:
                    text_dict[text1.tag] = text1.text.replace('\n', ' ').strip()
                problem['text'] = text_dict
            elif information.tag == 'quote':
                pass
            else:
                problem[information.tag] = information.text.replace(' ', '')
        problems.append(problem)
    return problems


'''
After get statement and candidate, we need to find the key word
which must distinguish candidate A and candidate B, which we need
drive the key-word
'''
def parse_candidate(candidateA, candidateB, pdps_nlp):
    #parse candidateA
    doc_A = pdps_nlp(candidateA)
    doc_B = pdps_nlp(candidateB)
        
    if doc_A.__len__ == 1:
        pre_candidateA = doc_A[0]
    else:
        for token in doc_A:
            if token.dep_ == 'ROOT':
                pre_candidateA = token
                break

    if doc_B.__len__ == 1:
        pre_candidateB = doc_B[0]
    else:
        for token in doc_B:
            if token.dep_ == 'ROOT':
                pre_candidateB = token
                break

    if pre_candidateA.text == pre_candidateB.text:
        for token in doc_A:
            if token.head.dep_ =='ROOT' and (token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
                pre_candidateA = token
        for token in doc_B:
            if token.head.dep_ =='ROOT' and (token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
                pre_candidateB = token

    return pre_candidateA, pre_candidateB

'''
After drive the key word of each candidate, we need embed those key
word into problem statement, so we could access dependency parse,
pos in the doc_statement as well as the doc_tensor
'''
def embed_candidate(pre_candidateA, pre_candidateB, pronoun,doc_state):
    #find pron in context
    for token in doc_state:
        if token.text.lower() == pronoun.lower():
            token_pronoun = token
    #find candidate A in context
    for token in doc_state:
        if pre_candidateA.lemma == token.lemma:
            token_candidateA = token
            break
    if token.i == doc_state.__len__() - 1:
        similarity = 0
        for token_2 in doc_state:
            if token_2.similarity(pre_candidateA) > similarity:
                similarity = token_2.similarity(pre_candidateB)
                token_candidateA = token_2
    #find candidate B in context
    for token in doc_state:
        if pre_candidateB.lemma == token.lemma:
            token_candidateB = token
            break
    if token.i == doc_state.__len__() - 1:
        similarity = -100
        for token_2 in doc_state:
            if token_2.similarity(pre_candidateB) > similarity:
                similarity = token_2.similarity(pre_candidateB)
                token_candidateB = token_2

    return token_candidateA, token_candidateB, token_pronoun

'''
Use tensor to do similarity estimate 

Because spaCy uses a 4-layer convolutional network to processing doc,
spacy will ncodes a document's internal meaning representations as an 
array of floats, also called a tensor. The tensors are sensitive to 
up to four words on either side of a word.
'''

def tensor_similarity(token_candidate, token_pronoun, doc):
    vector_candidate = doc.tensor[token_candidate.i]
    vector_pron = doc.tensor[token_pronoun.i]

    norm_candidate = np.linalg.norm(vector_candidate)
    norm_pron =np.linalg.norm(vector_pron)
  
    similarity = np.dot(vector_candidate, vector_pron)/(norm_candidate * norm_pron)
    return similarity


'''
Unsupervised semantic similarity method (USSM) as the first slover
In this function import doc, candidate and pron span, calcalate the
semantic sililarity between candidate and pron. By using FOFE encoding
algorithm (fixed-size ordinally-forgetting encoding)

'''
def calculate_vector(start, end, mid, doc, forgetting_factor):
    length = doc[0].vector.size
    vector = np.zeros((length,), dtype=np.float)
    for token in doc[start:end]:
        if token.i != mid:
            vector = forgetting_factor * np.array(vector) + token.vector
    return vector

def calculate_similarity(vector1, vector2):
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    similarity = np.dot(vector1, vector2)/(norm_vector1 * norm_vector2)

    return similarity

def USSM_solver(token_candidateA, token_candidateB, token_pronoun, doc_state, forgetting_factor = 0.5):
    candidate_semantic_vectorA = []
    candidate_semantic_vectorB = []
    pronoun_semantic_vector = []

    startA = max(token_candidateA.i - 2, 0)
    endA = min(token_candidateA.i + 2, doc_state.__len__())

    startB = max(token_candidateB.i - 2, 0)
    endB = min(token_candidateB.i + 2, doc_state.__len__())

    startP = max(token_pronoun.i - 1, 0)
    endP = min(token_pronoun.i + 4, doc_state.__len__())
    
    #calcalate vector of each semantic
    candidate_semantic_vectorA =  calculate_vector(startA, endA, token_candidateA.i, doc_state, forgetting_factor)
    candidate_semantic_vectorB =  calculate_vector(startB, endB, token_candidateB.i, doc_state, forgetting_factor)
    pronoun_semantic_vector =  calculate_vector(startP, endP, token_pronoun.i, doc_state, forgetting_factor)

    #calculate similarity
    SsimilarityA = calculate_similarity(candidate_semantic_vectorA, pronoun_semantic_vector)
    SsimilarityB = calculate_similarity(candidate_semantic_vectorB, pronoun_semantic_vector)

    return SsimilarityA, SsimilarityB

'''
How can we use commonsense knowledge to parse the lauguage
# def word_embedding_with_knowledge():
# https://github.com/iunderstand/SWE
learn from the conceptNET and try to understand the relation rule-base 
'''
def analysis_commensense(token_candidateA, token_candidateB, token_pronoun, doc_state):
    
    answer = None

    for token in doc_state[:token_pronoun.i]:
        if token.dep_ == 'ROOT' or token.dep_ == 'xcomp' or token.dep_ == 'ccomp' \
            or token.dep_ == 'advcl':
            key1_token = token
    assert(key1_token)

    for token in doc_state[token_pronoun.i:]:
        if token.dep_ == 'advcl' or token.dep_ == 'relcl' or token.dep_ == 'acomp' \
            or token.dep_ == 'ROOT' or token.dep_ == 'conj' or token.dep_ == 'ccomp' \
            or token.dep_ == 'xcomp':
            key2_token = token
    assert(key2_token)
    
    log ='Use Key words :' + key1_token.text + ' ' + key2_token.text + '\n'
    f.write(log)
    # Case for key1 and key2 are different type(verb and verb)
    if key1_token.pos_ == 'VERB' and key2_token.pos_ == 'VERB':
        relations = cn.relation(key1_token.lemma_, key2_token.lemma_)
        if 'Antonym' in relations:
            answer = 'B'
        elif 'Synonym' in relations:
            answer = 'A'
    # Another case for key1 and key2 are different type(verb and adj)
    elif  key1_token.pos_ == 'VERB' and key2_token.pos_ == 'ADJ':
        relations = cn.relation(key1_token.lemma_, key2_token.lemma_)
        if 'MotivatedByGoal' in relations:
            answer = 'A'
        elif 'RelatedTo' in relations:
            answer = 'B'

    return answer

def pdps_solver(statement, candidateA, candidateB, pronoun, pdps_nlp):
    
    #Answer metrics
    coreferenceA = dict()
    coreferenceB = dict()

    #use space pdps_nlp process to do pipline parse including tokenizer, POS, and dependency parse
    doc_state = pdps_nlp(statement)

    token_pronoun = None
    token_candidateA = None
    token_candidateB = None
        
    #parse candidade 
    pre_candidateA, pre_candidateB = parse_candidate(candidateA, candidateB, pdps_nlp)

    assert(pre_candidateA)
    assert(pre_candidateB)

    #embed the key word in the context
    token_candidateA, token_candidateB, token_pronoun = embed_candidate(pre_candidateA, pre_candidateB, pronoun, doc_state)

    assert(token_candidateA)
    assert(token_candidateB)
    assert(token_pronoun)

    print(token_candidateA, token_candidateB, token_pronoun)
    '''
    there is a new feature in spacy 2.0.0, call doc's tensor. Which represent meanings
    of words in context,As the processing pipeline is applied spaCy encodes a document's
    internal meaning representations as an array of floats, also called a tensor. This 
    allows spaCy to make a reasonable guess at a word's meaning, based on its surrounding
    words. 
    '''
    #here I will use tensor in doc to compute some Coreference between pron and candidate
    Tsimilarity_A = tensor_similarity(token_candidateA, token_pronoun, doc_state)
    Tsimilarity_B = tensor_similarity(token_candidateB, token_pronoun, doc_state)
        
    #print(Tsimilarity_A, Tsimilarity_B)
    coreferenceA['Tsimilarity'] = Tsimilarity_A
    coreferenceB['Tsimilarity'] = Tsimilarity_B

    assert(token_candidateB)

    #here I will use Unsupervised semantic similarity method to compute some Coreference between pron and candidate
    Ssimilarity_A, Ssimilarity_B = USSM_solver(token_candidateA, token_candidateB, token_pronoun, doc_state, forgetting_factor = 0.5)

    #print(Ssimilarity_A, Ssimilarity_B)
    coreferenceA['Ssimilarity'] = Ssimilarity_A
    coreferenceB['Ssimilarity'] = Ssimilarity_B

    '''
    learn from the conceptNET and try to understand the relation rule-base 
    '''
    if token_candidateA.dep_ == 'nsubj' and token_candidateB.dep_ == 'dobj' and token_pronoun.dep_ == 'nsubj':
        answer = analysis_commensense(token_candidateA, token_candidateB, token_pronoun, doc_state)
    elif token_candidateA.dep_ == 'nsubj' and token_candidateB.dep_ == 'pobj' and token_pronoun.dep_ == 'nsubj':
        answer = analysis_commensense(token_candidateA, token_candidateB, token_pronoun, doc_state)
    else:
        f.writelines('Use Similarity \n')
        answer = None

    if answer == None:
        possibilityA = Ssimilarity_A + Tsimilarity_A
        possibilityB = Ssimilarity_B + Tsimilarity_B
        if possibilityA >= possibilityB:
            answer = 'A'
        else:
            answer = 'B'
    
    return answer

if __name__ == "__main__":
    
    # parse the xml to problem we want.
    print('starting parse the xml file......\n')
    problems = wsc_pdps_input_data()
    print('finished parse \n')

    # load spacy language as nlp 
    print('starting loading spacy model en_core_web_sm..... \n')
    pdps_nlp =  spacy.load('en_core_web_sm')
    print('finished loading spacy model en_core_web_sm \n')
    print('------------------------------------------------')
    #print out the pipline name we use in spacy tool 
    print(pdps_nlp.pipe_names)

    #conceptNet API it's how I get commonsense knowledge
    cn = conceptNet()

    #  
    f = open('Result.txt', 'w')    
    problem_num = 0
    right_num = 0

    for problem in problems:
        #correct answer
        correct_answer = problem['correctAnswer']
        if len(correct_answer) != 1:
            correct_answer = correct_answer[0]
        #drive problem statement
        statement = problem['text']['txt1'] + ' ' + problem['text']['pron'] + ' ' + problem['text']['txt2']
        #drive two candidate and the pronoun we need understand
        candidateA = problem['answers'][0]
        candidateB = problem['answers'][1]
        pronoun = problem['text']['pron']

        problem_num = problem_num + 1

        print('Problem Number: ', problem_num)
        log = 'Problem Number: ' +  str(problem_num) + '\n'
        f.write(log)
        
        answer = pdps_solver(statement, candidateA, candidateB, pronoun, pdps_nlp)

        log = 'answer : ' + answer +' correctAnswer: ' + correct_answer + '\n'
        f.write(log)

        if answer == correct_answer:
            right_num = right_num + 1

    #calculate the ratio of correct answer
    correct_ration = right_num * 1.0 / problem_num
    print(correct_ration)
    f.close()