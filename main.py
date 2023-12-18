from typing import Dict

from openfabric_pysdk.utility import SchemaUtil
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

from model.ChatbotModel import ChatbotModel

# Initialize the chatbot model with the specified path
model_path = "./model/chatbot"
chatbot = ChatbotModel(model_path)


############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    """
    Callback function called on updating configurations.

    Args:
    - configuration: Dictionary containing configuration values.
    - state: State object for maintaining state information.

    Note: Modify this function to handle configuration changes if needed.
    """
    pass

############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    """
    Callback function called for each execution pass.

    Args:
    - request: SimpleText object containing input text.
    - ray: Ray object for communication between tasks.
    - state: State object for maintaining state information.

    Returns:
    - SimpleText object containing generated responses.

    Note: Adjust this function based on your specific use case and requirements.
    """
    output = []

    # Check if input text exists
    if not request or not request.text:
        raise ValueError("Input text is empty or does not exist")

    # Loop through input texts and generate responses
    for text in request.text:
        try:
            # Generate response for each input text
            response = chatbot.generate_answer(text, max_length=120)
            print(response)
            output.append(response)

        except Exception as e:
            # Log error message
            error_msg = f"Error generating response for input text: {text}. Error: {e}"
            print(error_msg)

    # Return the output as a SimpleText object
    return SchemaUtil.create(SimpleText(), dict(text=output))
