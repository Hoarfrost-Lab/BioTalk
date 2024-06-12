
from utils import utils
import torch
from transformers import RobertaModel, AutoTokenizer, AutoModel, AutoModelForMaskedLM

def main():
    train_df = utils.load_df('/work/ah2lab/soumya/','SwissprotDatasets/UnbalancedSwissprot/train')
    test_df = utils.load_df('/work/ah2lab/soumya/','SwissprotDatasets/test')
    train_sequence = train_df['Sequence'].tolist() #[::10]
    train_EC = train_df['EC'].tolist()#[::10]
    test_sequence = test_df['Sequence'].tolist()
    test_EC = test_df['EC'].tolist()
    batch_size = 512

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    # model = AutoModelForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = RobertaModel.from_pretrained("models/LOLBERTv4/")
    tokenizer = utils.load_tokenizer("LOLBERTv4/") #FinetunedModel LOLBERTv4
    # tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", trust_remote_code=True, local_files_only=True)
    # model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", trust_remote_code=True, local_files_only=True)
    model.to(device)
    device_ids = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    train_embeddings =  utils.get_embeddings(tokenizer, model, device, train_sequence, batch_size)
    test_embeddings =  utils.get_embeddings(tokenizer, model, device, test_sequence, batch_size)

    train_embedding_stats = get_embedding_stats(train_embeddings)
    test_embedding_stats = get_embedding_stats(test_embeddings)
    parameters = ['class', 'min', 'max', 'avg']
    K = [1, 3, 5]
    for train_stat, test_stat, parameter in zip(train_embedding_stats, test_embedding_stats, parameters):
        accuracy = utils.top_k_retrieval(train_EC, test_EC, train_stat, test_stat, K)
        print(f"For {tokenizer.name_or_path}'s {parameter.upper()} nearest retrieval stats {accuracy}")


if __name__ == "__main__":
    main()
