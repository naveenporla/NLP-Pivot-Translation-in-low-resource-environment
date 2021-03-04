import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from transformer_utils import bleu, Translate,Translate_testdata,cust_bleu
from torchtext.data import Field, BucketIterator,TabularDataset
import numpy as np
from tqdm import tqdm


checkpoint_path = 'Models/Tanzil/100k/tf_az_en_100k.pth.tar'                ##To save model which obtained minimum validation loss
each_checkpoint = "Models/Tanzil/100k/each_tf_az_en_100k.pth.tar"           ##To save model after every epoch in google colab
max_bleu_checkpoint_path = "Models/Tanzil/100k/max_tf_az_en_100k.pth.tar"   ##To save model which got max bleu score during training
train_dataset_path = 'Datasets/Tanzil/az_en_train100k.csv'
test_dataset_path = 'Datasets/Tanzil/az_en_test100k.csv'
set_source_to = "azerbaijani"     ##to setup tokenizer
set_target_to = "english"     ##to setup tokenizer

#
#checkpoint_path = 'Models/Tanzil/200k/tf_az_tr_200k.pth.tar'
#each_checkpoint = "Models/Tanzil/200k/each_tf_az_tr_200k.pth.tar"
#max_bleu_checkpoint_path = "Models/Tanzil/200k/max_tf_az_tr_200k.pth.tar"
#train_dataset_path = 'Datasets/Tanzil/tr_az_train200k.csv'
#test_dataset_path = 'Datasets/Tanzil/tr_az_test200k.csv'
#set_source_to = "azerbaijani"     #to setup tokenizer
#set_target_to = "turkish"     #to setup tokenizer
#

checkpoint_path = "Models/Tanzil/200k/tf_tr_en_200k.pth.tar"
each_checkpoint = "Models/Tanzil/200k/each_tf_tr_en_200k.pth.tar"
max_bleu_checkpoint_path = "Models/Tanzil/200k/max_tf_tr_en_200k.pth.tar"
train_dataset_path = 'Datasets/Tanzil/tr_en_train200k.csv'
test_dataset_path = 'Datasets/Tanzil/tr_en_test200k.csv'
set_source_to = "turkish"     #to setup tokenizer
set_target_to = "english"     #to setup tokenizer


#turk-eng:
#sentence = "Allaha ve Peygamberine kim inanmamışsa bilsin ki şüphesiz Biz inkarcılar için çılgın alevli cehennemi hazırlamışızdır"
#sentence_eng = "And whoever does not accept faith in Allah and His Noble Messenger – We have indeed kept prepared a blazing fire for disbelievers"


#azer-turkish
#sentence = "O yer üzünü sizin üçün beşik etmiş orada sizin üçün yollar salmış və göydən su endirmişdir"
#sentence_turk = "O ki yeri size beşik yaptı ve onda sizin için yollar açtı gökten bir su indirdi"



#sentence = sentence.split()
#python -m spacy download en
spacy_eng = spacy.load("en")


tokenize_custom = lambda x: x.split()

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


src_tokenizer = None
trg_tokenizer = None

if set_source_to == "azerbaijani" or set_source_to == "turkish":
    src_tokenizer = tokenize_custom
elif set_source_to == "english":
    src_tokenizer = tokenize_eng

if set_target_to == "azerbaijani" or set_target_to == "turkish":
    trg_tokenizer = tokenize_custom
elif set_target_to == "english":
    trg_tokenizer = tokenize_eng

source_lang = Field(tokenize=src_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>")

target_lang = Field(tokenize=trg_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>")



fields = {'Source_lang': ('src',source_lang), 'Target_lang' : ('trg',target_lang)}

train_data, test_data = TabularDataset.splits(
        path='',
        train = train_dataset_path,
        test = test_dataset_path,
        format = 'csv',
        fields = fields
        )

source_lang.build_vocab(train_data, max_size=20000, min_freq=2)
target_lang.build_vocab(train_data, max_size=20000, min_freq=2)


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("device set to:",device)


# Training hyperparameters
num_epochs = 3
patience = 10

learning_rate = 0.0005 #0.0003
batch_size = 32

# Model hyperparameters
src_vocab_size = len(source_lang.vocab)
trg_vocab_size = len(target_lang.vocab)
embedding_size = 200 #512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 300
forward_expansion = 512
src_pad_idx = target_lang.vocab.stoi["<pad>"]


train_iterator,test_iterator = BucketIterator.splits(
    (train_data,test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,mode = 'min', factor=0.1, patience=10, verbose=True
)

pad_idx = target_lang.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


#sentence = "Bu Allah tərəfindən nazil edilməsinə haqdan gəlməsinə heç bir şəkkşübhə olmayan müttəqilərə Allahdan qorxanlara pis əməllərdən çəkinənlərə doğru yol göstərən Kitabdır"
#sentence ="This is The Book free of doubt and involution a guidance for those who preserve themselves from evil and follow the straight path" 

#sentence = sentence.split()

#Early Stopping variables below
training_loss = []


Max_BleuScore = None
Max_Testloss = None

counter = 0

avg_trainloss_plot = []
Bleu_scores_plot = []


for epoch in tqdm(range(num_epochs), desc="Epoch Loop "):
    print('[Epoch {} / {}], [lr = {}]'.format(epoch,num_epochs, optimizer.param_groups[0]['lr']))

    model.train()
    losses = []
    test_losses = []

    for batch_idx, batch in enumerate(train_iterator):
        #print("batch_idx",batch_idx)

        inp_data = batch.src.to(device)
        target = batch.trg.to(device)


        output = model(inp_data, target[:-1, :])

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()
    
    avg_train_loss = np.average(losses)
    avg_trainloss_plot.append(avg_train_loss)
    model.eval()
    
    for t_batch_idx, test_batch in enumerate(test_iterator):
        #print("batch_idx",batch_idx)

        test_inp_data = test_batch.src.to(device)
        test_target = test_batch.trg.to(device)

        # Forward prop
        #print("target-->",target)
        with torch.no_grad():
            test_output = model(test_inp_data, test_target[:-1, :])

        test_output = test_output.reshape(-1, test_output.shape[2])
        test_target = test_target[1:].reshape(-1)

        test_loss = criterion(test_output, test_target)
        test_losses.append(test_loss.item())
        
    avg_test_loss = np.average(test_losses)
    
    
    ######Uncomment below code to see how translation is working during training, give only one sentence#####
    
#    ##translated_sentence = Translate(
#    ##    model, sentence, source_lang, target_lang, device, max_length=50
#    ##)
#    ##print(f"Translated example sentence: \n {translated_sentence}")
    
    print(f"\nsaving model after epoch: {epoch}")
    torch.save(model,each_checkpoint)
    
    
    BleuScore = bleu(test_data[:100], model, source_lang, target_lang, device,max_length=50)
    Bleu_scores_plot.append(BleuScore)
    
    print(f'\n[epoch:{epoch}] -->>'+f'training loss is: {avg_train_loss:.5f}'+f' test loss is: {avg_test_loss:.5f}' + f' Bleu Score is: {BleuScore*100:.5f}')
   
    if Max_BleuScore == None:
        Max_BleuScore = BleuScore
        print("New max bleu score found , saving model")
        torch.save(model,max_bleu_checkpoint_path)
    elif BleuScore > Max_BleuScore:
        Max_BleuScore = BleuScore
        print("New max bleu score found , saving model")
        torch.save(model,max_bleu_checkpoint_path)
   #Earlystop check
    #BleuScore = avg_test_loss
    
    if Max_Testloss == None:
        Max_Testloss = avg_test_loss
       #print("First min loss",min_loss)
    elif avg_test_loss < Max_Testloss:
        Max_Testloss = avg_test_loss
        print("New min loss found , saving model")
        torch.save(model,checkpoint_path)
        counter = 0
    else:
        counter +=1
        print("counter is:",counter)
        if counter == patience:
            print("early stopping as patience is reached")
            break
   
   ###saving the last epoch model just in case

    
    #mean_loss = sum(losses) / len(losses)
    scheduler.step(avg_test_loss)

print("training finished")

model = torch.load(each_checkpoint)
### running on entire test data takes a while
score = bleu(test_data, model, source_lang, target_lang, device,max_length=50)
print(f"Bleu score {score * 100:.2f}")



#### following code is useful to load the models and do pivot translation #####

#translation_input_path = "Testset/Tanzil/azer_20k.txt"
##translation_input_path = "Outputs/Tanzil/pivot_turk_translated_2.txt"
#translation_output_path = "Outputs/Tanzil/tf_az_en_eng_20k_test.txt"
#
#Translate_testdata(translation_input_path,translation_output_path,model, source_lang, target_lang, device,max_length=50)
#
#bleu_Score = cust_bleu(translation_output_path, "Testset/Tanzil/eng_20k_test.txt")
#print(f"Custom Bleu score Transformer: {bleu_Score*100:.3f}")
#


######  Training loss and Bleu plot data ######
#with open("Outputs/Tanzil/az_en_plotdata100k.txt", 'w',encoding='utf-8') as op:
#    for i in range(num_epochs):
#        op.write(f"{i}"+' '+f"{avg_trainloss_plot[i]:.4f}"+' '+f"{Bleu_scores_plot[i]*100:.4f}"+'\n')

print("Finished")
