from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import Dataset, load_dataset


class CustomData():
    def __init__(self, data_file="data/data.csv", split="train"):
        
        """
            Initialize the dataset:
            - Load the CSV file
            - Initialize the tokenizer
            - Prepare the tokenized dataset
        """

        # Load pre-trained DistilBERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Load the raw dataset from CSV
        self.raw_data = load_dataset('csv', data_files=data_file)
        
        # Store which split to use (train/test/validation)
        self.split_name = split
        
        # Tokenize and clean the dataset
        self.final_dataset = self.map_and_clean_data()
    
    def tokenizer_data(self, data):

        """
        Tokenizes input questions and target answers.
        
        Args:
            data (dict): A batch of examples containing 'question' and 'answer'.
        
        Returns:
            dict: Tokenized inputs with 'input_ids', 'attention_mask', and 'labels'.
        """

        # Tokenize questions (inputs for the model)
        model_inputs = self.tokenizer(
            data['question'],
            padding=True,
            truncation=True
        )

        # Tokenize answers as target labels
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(data['answer'], padding=True, truncation=True)
        
        # Add labels to the model inputs
        model_inputs['labels'] = labels['input_ids']
        
        return model_inputs
    
    def map_and_clean_data(self):

        """
        Apply tokenization to the dataset and remove unnecessary columns.
        
        Returns:
            Dataset: A cleaned and tokenized dataset ready for PyTorch DataLoader.
        """

        # Apply tokenization function to all examples in batches
        tokenizer_dataset = self.raw_data.map(self.tokenizer_data, batched=True)
        
        # Select the specified split
        data_split = tokenizer_dataset[self.split_name]
        
        # Remove raw text columns to keep only tensors for training
        dataset = data_split.remove_columns(['question', 'answer'])
        return dataset
    
    def get_dataloader(self, batch_size=8, shuffle=True):

        """
        Returns a PyTorch DataLoader for the tokenized dataset.
        
        Args:
            batch_size (int): Number of examples per batch
            shuffle (bool): Whether to shuffle the dataset
        
        Returns:
            DataLoader: PyTorch DataLoader ready for training
        """

        # Automatically pad sequences in a batch
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Format dataset to return PyTorch tensors
        self.final_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

        # Create and return the DataLoader
        return DataLoader(
            self.final_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator
        )
