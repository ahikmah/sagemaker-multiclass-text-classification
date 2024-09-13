import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd

dataset_path = 's3://hugging-face-multiclass-bucket/training_data/newsCorpora.csv'

df = pd.read_csv(dataset_path, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

df = df[['TITLE', 'CATEGORY']]

my_dict = {
    'e': 'Entertainment',
    'b': 'Business',
    't': 'Science',
    'm': 'Health'
}

def update_category(x):
    return my_dict[x]

df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_category(x))

print(df)

# This is just a tip
# Test the model if its working properly. Train the model with the small subset of data, because the training will take many many hours and it will be super expensive

# df = df.sample(frac=0.05, random_state=1)

# df = df.reset_index(drop=True)

# This is where the tip ends


encode_dict = {}

def encode_category(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

df['ENCODE_CATEGORY'] = df['CATEGORY'].apply(lambda x: encode_category(x))

df = df.reset_index(drop=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        
    def __getitem__(self, index):
        title = str(self.data.iloc[index, 0])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_len=self.max_len
            padding='max-length',
            return_token_type_ids=True,
            truncation=True,
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids':torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index, 2], dtype=torch.long)
        }
    
    def __len__(self):
        return self.len

    
train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

train_dataset.reset_index(drop=True)


print("Full dataset: {}".format(df.shape))
print("Train dataset: {}".format(train_dataset.shape))
print("Test dataset: {}".formt(test_dataset.shape))



MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2

training_set = NewsDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = NewsDataset(test_dataset, tokenizer, MAX_LEN)

train_parameters = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0,
}

test_parameters = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0,
}


training_loader = DataLoader(training_set, **train_parameters)
testing_loader = DataLoader(testing_set, **test_parameters)

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768) #transform the feature vector from the DistilBERTModel
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4) #4 possible classes for the output (prop score)
    
    def forward(self, input_ids, attention_mask):
        '''
            If you were to initialize the model’s layers inside the forward method, 
            it would reset the weights every time a forward pass is executed.
            This would mean that the model would never actually learn anything
            because the weights would be reinitialized to their starting values on each pass.
        '''
        output_1 = self.l1(input_ids = input_ids, attention_mask=attention_mask)
                
        hidden_state = output_1[0] 
        
        pooler = hidden_state[:,0]
        
        pooler = self.pre_classifier(pooler)
        
        pooler = torch.nn.ReLU()(pooler)
        
        pooler = self.dropout(pooler)
        
        output = self.classifier(pooler)
        
        return output
    

def calculate_accuracy(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    # print(big_idx == targets).sum() #tensor(2)
    # print(big_idx == targets).sum().item() 2
    
    return n_correct


def training(epoch, model, device, training_loader, optimizer, loss_function):
    tr_loss=0
    n_correct=0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accuracy(big_idx, targets)
        
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        
        if _%5000 == 0:
            loss_step = tr_loss/nb_tr_steps
            accuracy_step = (n_correct*100)/nb_tr_examples
            print(f"Training loss per 5000steps: {loss_step}")
            print(f"Training accuracy oer 5000 steps: {accuracy_step}")
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    # print(f"The total accuracy for epoch {epoch}: {(n_correct*100)/nb_tr_examples}")
    epoch_loss = train_loss / nb_tr_steps
    epoch_accuracy =  (n_correct*100)/nb_tr_examples
    
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training accuracy Epoch: {epoch_accuracy}")
    
    
    return


def validation(epoch, model, testing_loader, device, loss_function):
    
    model.eval()
    
    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    
    with torch.no_grad():    
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs  =model(ids, mask).squeeze()

            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)


            if _ % 1000 == 0:
                loss_step = tr_loss/nb_tr_steps
                accuracy_step = (n_correct*100)/nb_tr_examples
                print(f"Validation loss per 1000 steps: {loss_step}")
                print(f"Validation accuracy per 1000 steps: {epoch_accuracy}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accuracy =  (n_correct*100)/nb_tr_examples
    print(f"Validation loss per Epoch: {epoch_loss} at epoch {epoch}")
    print(f"Validation accuracy per epoch {epoch_accuracy} at epoch {epoch}")
    
    return
    
    
def main():
    print("start")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size",type=int,default=2)
    parser.add_argument("--learning_rate",type=float,default=5e-5)
    
    args = parser.parse_args()
    
    args.epochs
    args.train_batch_size
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    model = DistilBERTClass()
    
    model.to(device)
    
    LEARNING_RATE = 1e-05
    optimizer = torch.optim.Adam(params = model.parameters(, lr=LEARNING_RATE))
    
    loss_function = torch.nn.CrossEntropy()
    
    # Train loop
    
    EPOCHS = 2
    
    for epoch in range(EPOCHS):
        print(f"starting epoch: {epoch}")
        
        training(epoch, model, device, training_loader, optimizer, loss_function)
        
        validation(epoch, model, testing_loader, device, loss_function)
    
    output_dir = os.environ['SM_MODEL_DIR']
    
    output_model_file = os.path.join(output_dir, 'pytorch_distilbert_news.bin')
    
    output_vocab_file = os.path.join(output_dir, 'vocab_distilbert_news.bin')
    
    torch.save(model.state_dict(), output_model_file)
    
    tokenizer.save_vocabulary(output_vocab_file)
    
    
if __name__ == '__main__':
    main()