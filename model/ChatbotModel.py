import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ChatbotModel:
    """
    A class responsible for generating responses using a fine-tuned GPT-2 model.

    Attributes:
    - model_path: str, path to the directory containing the GPT-2 model.
    """

    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        """
        Load the GPT-2 model.

        Returns:
        - model: GPT2LMHeadModel, the GPT-2 model.
        """
        model = GPT2LMHeadModel.from_pretrained(self.model_path)
        return model

    def load_tokenizer(self):
        """
        Load the GPT-2 tokenizer.

        Returns:
        - tokenizer: GPT2Tokenizer, the GPT-2 tokenizer.
        """
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        return tokenizer

    def process_output(self, text):
        """
        Process the output of the model.

        Args:
          output: The output of the model.

        Returns:
          The processed output.
        """
        match = re.search(r'"([^"]*)"', text)

        if match:
            extracted_sentence = match.group(1)
            return extracted_sentence

        return text

    def generate_answer(self, question, max_length):
        """
        Generate an answer to a given question using the GPT-2 model.

        Args:
        - question: str, the input question.
        - max_length: int, the maximum length of the generated answer.

        Returns:
        - answer: str, the generated answer.
        """
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        ids = tokenizer.encode(f'{question}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )
        return self.process_output(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
