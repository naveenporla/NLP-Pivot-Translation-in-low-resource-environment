import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def Translate(model, sentence, source_lang, target_lang, device, max_length):
    if(len(sentence) == 0):
        return ""
    # Add <SOS> and <EOS> in beginning and end respectively
    if(sentence[0] != "<sos>"):
        sentence.insert(0, source_lang.init_token)
        sentence.append(source_lang.eos_token)
    
#    sentence.insert(0, source_lang.init_token)
#    sentence.append(source_lang.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [source_lang.vocab.stoi[token] for token in sentence]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [target_lang.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == target_lang.vocab.stoi["<eos>"]:
            break

    translated_sentence = [target_lang.vocab.itos[idx] for idx in outputs]
    return translated_sentence


def bleu(data, model, source_lang, target_lang, device,max_length):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = Translate(model, src, source_lang, target_lang, device,max_length)
        prediction = prediction[1:-1]  # remove <sos> <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def Translate_testdata(testdata_path,output_path, model, source_lang, target_lang, device, max_length):
    src_data = open(testdata_path, encoding='utf8').read().lower().split('\n')
    outputs = []
    for row in src_data:
        src = []
        src = row.split(' ')
        prediction = Translate(model, src, source_lang, target_lang, device,max_length)
        prediction = prediction[1:-1]
        outputs.append(prediction)
    
    writer = [' '.join(s) for s in outputs]
    with open(output_path, 'w',encoding='utf-8') as op:
            for sent in writer[:-1]:    
                op.write(sent + '\n')
            op.write(writer[-1])
    return

   
    
def cust_bleu(output_path,target_path):
    spacy_eng = spacy.load("en")
    
    
    output_data = open(output_path, encoding='utf8').read().split('\n')
    target_data = open(target_path, encoding='utf8').read().split('\n')    
    outputs = [s.lower().split(' ') for s in output_data]
    targets=[]
    for sent in target_data:
        targets.append([[tok.text.lower() for tok in spacy_eng(sent)]])
        
    #print("cust outputs-->",outputs)
    #print("cust tagets-->",targets)
    return bleu_score(outputs, targets)
