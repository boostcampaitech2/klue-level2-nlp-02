import re

def preprocess_sen(sen) :
    sen = re.sub('[^가-힣0-9a-zA-Z\[\]\',.!?]' , ' ', sen)
    sen = re.sub(' {2,}' , ' ' , sen)
    return sen