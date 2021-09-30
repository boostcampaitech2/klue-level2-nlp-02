import re

def preprocess_sen(sen) :
    sen = re.sub('[^가-힣0-9a-zA-Z\[\]\',.!?]@~*#' , ' ', sen)
    sen = re.sub(' {2,}' , ' ' , sen)
    return sen

def sentence_processing(data):
    new_sentence = []
    for row in data.values:
        sentence, subject_entity, object_entity = row[1], eval(row[2]), eval(row[3])
        sub_start_idx, sub_end_idx, sub_type = subject_entity['start_idx'], subject_entity['end_idx'], subject_entity['type']
        ob_start_idx, ob_end_idx, ob_type = object_entity['start_idx'], object_entity['end_idx'], object_entity['type']
        
        if sub_start_idx < ob_start_idx :
            sentence = sentence[:sub_start_idx] + ' @ * ' + sub_type + ' * ' + sentence[sub_start_idx:sub_end_idx+1] + ' @ ' + sentence[sub_end_idx+1:ob_start_idx] + ' # ~ ' + ob_type + ' ~ ' + sentence[ob_start_idx:ob_end_idx+1] + ' # ' + sentence[ob_end_idx+1:]
        else :
            sentence = sentence[:ob_start_idx] + ' # ~ ' + ob_type + ' ~ ' + sentence[ob_start_idx:ob_end_idx+1] + ' # ' + sentence[ob_end_idx+1:sub_start_idx] + ' @ * ' + sub_type + ' * ' + sentence[sub_start_idx:sub_end_idx+1] + ' @ ' + sentence[sub_end_idx+1:]
        
        sentence = re.sub('\s+', " ", sentence)
        new_sentence.append(sentence)
    
    return new_sentence   