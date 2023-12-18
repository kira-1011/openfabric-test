from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer

class ChatbotTrainer:
    """
    A class responsible a Fine tuning GPT2 using custom dataset.

    Attributes:
    - train_file_path: str, path to the training dataset file in CSV format.
    - model_name: str, the name of the GPT-2 model to use.
    - output_dir: str, path to the directory where the trained model will be saved.
    - overwrite_output_dir: bool, whether to overwrite the output directory if it exists.
    - per_device_train_batch_size: int, the training batch size per device.
    - num_train_epochs: float, the number of training epochs.
    - save_steps: int, the number of training steps before saving the model.
    """

    def __init__(self, train_file_path, model_name, output_dir, overwrite_output_dir,
                 per_device_train_batch_size, num_train_epochs, save_steps):
        self.train_file_path = train_file_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.overwrite_output_dir = overwrite_output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.save_steps = save_steps

    def load_dataset(self, tokenizer, block_size=128):
        """
        Load the training dataset.

        Args:
        - tokenizer: GPT2Tokenizer, the tokenizer used for tokenizing the dataset.
        - block_size: int, the maximum block size for each training example.

        Returns:
        - dataset: TextDataset, the training dataset.
        """
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=self.train_file_path,
            block_size=block_size,
        )
        return dataset

    def load_data_collator(self, tokenizer, mlm=False):
        """
        Load the data collator for language modeling.

        Args:
        - tokenizer: GPT2Tokenizer, the tokenizer used for tokenizing the dataset.
        - mlm: bool, whether to use the Masked Language Modeling (MLM) objective.

        Returns:
        - data_collator: DataCollatorForLanguageModeling, the data collator.
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
        )
        return data_collator

    def train(self):
        """
        Train the GPT-2 model.
        """
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        train_dataset = self.load_dataset(tokenizer)
        data_collator = self.load_data_collator(tokenizer)

        tokenizer.save_pretrained(self.output_dir)

        model = GPT2LMHeadModel.from_pretrained(self.model_name)

        model.save_pretrained(self.output_dir)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=self.overwrite_output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            num_train_epochs=self.num_train_epochs,
            save_steps=self.save_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
        trainer.save_model()
