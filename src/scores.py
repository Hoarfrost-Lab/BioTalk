
from utils import utils
import torch
from transformers import RobertaModel, AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sklearn import metrics
def main():
    train_df = utils.load_df('/work/ah2lab/soumya/','UniprotDatasets/BalancedUniprot/Train')
    print(len(train_df))
    filtered_df = train_df.groupby('EC').head(20) #.sort_values('EC')
    print(len(train_df), len(filtered_df))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    # model = AutoModelForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = RobertaModel.from_pretrained("models/FinetunedModel/")
    tokenizer = utils.load_tokenizer("FinetunedModel/") #FinetunedModel LOLBERTv4
    # tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", trust_remote_code=True, local_files_only=True)
    # model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", trust_remote_code=True, local_files_only=True)
    model.to(device)
    device_ids = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    batch_size = 512
    train_sequences = filtered_df['Sequence'].tolist()
    train_EC = filtered_df['EC'].tolist()
    stat = ['class', 'min', 'max', 'avg']
    train_embeddings =  utils.get_embeddings(tokenizer, model, device, train_sequences, batch_size)
    for name, item in zip(stat, utils.get_embedding_stats(train_embeddings)):
        print(f"silhouette_score of {tokenizer.name_or_path} for stat {name} = {metrics.silhouette_score(item, train_EC)}")
    
if __name__ == "__main__":
    main()
