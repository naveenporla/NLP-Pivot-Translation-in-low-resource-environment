import torch
from torchtext.data.metrics import bleu_score
import spacy

def Translate(model, sentence, source_lang, target_lang, device, max_length):
    if(len(sentence) == 0):
        return ""
    # Add <SOS> and <EOS> in beginning and end respectively
    if(sentence[0] != "<sos>"):
        sentence.insert(0, source_lang.init_token)
        sentence.append(source_lang.eos_token)

    # Convert sentence to tokens
    text_to_indices = [source_lang.vocab.stoi[token] for token in sentence]
    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)
        
    outputs = [target_lang.vocab.stoi["<sos>"]]

    for i in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            max_output = output.argmax(1).item()
        
        #print("max_output--->",max_output)
        
        outputs.append(max_output)
        # if model predicts end of sentence then break
        if max_output == target_lang.vocab.stoi["<eos>"] or max_output < 0 or max_output > len(target_lang.vocab):
            break

    translated_sentence = [target_lang.vocab.itos[idx] for idx in outputs]

    return translated_sentence




def bleu(data, model, source_lang, target_lang, device,max_length,generate_outputs):
    targets = []
    outputs = []
    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        #print("trg=========>",trg)
        #print("src==",src)
        prediction = Translate(model, src, source_lang, target_lang, device,max_length)
        #print("predictions===",prediction)
        # remove <sos> <eos> token
        prediction = prediction[1:-1]  

        targets.append([trg])
        outputs.append(prediction)
        
    if generate_outputs:
        writer = [' '.join(s) for s in outputs]
        with open("Outputs/testset_translated.txt", 'w',encoding='utf-8') as op:
            for sent in writer[:-1]:    
                op.write(sent + '\n')
            op.write(writer[-1])
    #print("org outputs-->",outputs)
    #print("org tagets-->",targets)
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
    
    
    
        
    