import boto3
import json
import os
from dispatchers import utils
# from dispatchers import bedrockutils 
import logging

logger = utils.get_logger(__name__)
logger.setLevel(logging.DEBUG)
CHAT_HISTORY="chat_history"
initial_history = {CHAT_HISTORY: f"AI: Hi there! How Can I help you?\nHuman: ",}

## bedrockutils

from typing import List, Any, Dict
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, ConversationChain
from langchain.llms.bedrock import Bedrock
# from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.schema import BaseMemory
# from pydantic import BaseModel, Extra
import json

class LexConversationalMemory(BaseMemory):

    """Langchain Custom Memory class that uses Lex Conversation history
    
    Attributes:
        history (dict): Dict storing conversation history that acts as the Langchain memory
        lex_conv_context (str): LexV2 sessions API that serves as input for convo history
            Memory is loaded from here
        memory_key (str): key to for chat history Langchain memory variable - "history"
    """
    history = {}
    memory_key = "chat_history" #pass into prompt with key
    lex_conv_context = ""

    def clear(self):
        """Clear chat history
        """
        self.history = {}

    @property
    def memory_variables(self) -> List[str]:
        """Load memory variables
        
        Returns:
            List[str]: List of keys containing Langchain memory
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load memory from lex into current Langchain session memory
        
        Args:
            inputs (Dict[str, Any]): User input for current Langchain session
        
        Returns:
            Dict[str, str]: Langchain memory object
        """
        input_text = inputs[list(inputs.keys())[0]]

        ccontext = json.loads(self.lex_conv_context)
        memory = {
            self.memory_key: ccontext[self.memory_key] + input_text + "\nAI: ",
        }
        return memory

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Load memory from lex + current input into Langchain session memory
        
        Args:
            inputs (Dict[str, Any]): User input
            outputs (Dict[str, str]): Langchain response from calling LLM
        """
        input_text = inputs[list(inputs.keys())[0]]
        output_text = outputs[list(outputs.keys())[0]]

        ccontext = json.loads(self.lex_conv_context)
        self.history =  {
            self.memory_key: ccontext[self.memory_key] + input_text + f"\nAI: {output_text}",
        }

class BedrockLangchainBot():

    """Create a langchain.ConversationChain using a Sagemaker endpoint as the LLM
    
    Attributes:
        chain (langchain.ConversationChain): Langchain chain that invokes the Sagemaker endpoint hosting an LLM
    """
    
    def __init__(self, prompt_template, 
        lex_conv_history="",
        region_name="" ):
        """Create a SagemakerLangchainBot client
        
        Args:
            prompt_template (str): Prompt template
            sm_endpoint_name (str): Sagemaker endpoint name
            lex_conv_history (str, optional): Lex convo history from LexV2 sessions API. Empty string for no history (first chat)
            region_name (str, optional): region where Sagemaker endpoint is deployed
        """


        prompt = PromptTemplate(
            input_variables=["chat_history", "input"],
            template=prompt_template
        )
        
        session = boto3.Session(region_name = 'us-east-1')
        boto3_bedrock = session.client(service_name="bedrock-runtime")

        # Sagemaker endpoint for the LLM. Pass in arguments for tuning the model and
        bedrock_llm = Bedrock(
            client=boto3_bedrock,
            region_name='us-east-1',
            model_id='anthropic.claude-v2:1'
            # model_kwargs={"temperature":2.0,"max_length":50, "num_return_sequences":3, "top_k":50, "top_p":0.95, "do_sample":True}
        )

        # Create a conversation chain using the prompt, llm hosted in Sagemaker, and custom memory class
        self.chain = ConversationChain(
            llm=bedrock_llm, 
            prompt=prompt, 
            memory=LexConversationalMemory(lex_conv_context=lex_conv_history), 
            verbose=True
        )

    def call_llm(self,user_input) -> str:
        """Call the Sagemaker endpoint hosting the LLM by calling ConversationChain.predict()
        
        Args:
            user_input (str): User chat input
        
        Returns:
            str: Sagemaker response to display as chat output
        """
        output = self.chain.predict(
            input=user_input
        )
        print("call_llm - input :: "+user_input)
        print("call_llm - output :: "+output)
        return output 


class QnABotBedrockLangchainDispatcher():
    
    def __init__(self, intent_request):
        # QnABot Session attributes
        self.intent_request = intent_request
        self.input_transcript = self.intent_request['req']['question']
        self.intent_name = self.intent_request['req']['intentname']
        self.session_attributes = self.intent_request['req']['session']

    def dispatch_intent(self):
        # prompt_template = """The following is a friendly conversation between a human and an AI. The AI is 
        # talkative and provides lots of specific details from its context. If the AI does not know 
        # the answer to a question, it truthfully says it does not know. You are provided with information
        # about entities the Human mentions, if relevant.

        # Chat History:
        # {chat_history}

        # Conversation:
        # Human: {input}
        # AI:"""

        # Claude specific
        prompt_template = """The following is a friendly conversation between a human and an AI. The AI is 
        talkative and provides lots of specific details from its context. If the AI does not know 
        the answer to a question, it truthfully says it does not know. You are provided with information
        about entities the Human mentions, if relevant.

        Chat History:
        {chat_history}

        Conversation:
        Human: {input}
        
        
        Assistant:"""

        if 'ConversationContext' in self.session_attributes:
            # Set context with convo history for custom memory in langchain
            conv_context: dict = self.session_attributes.get('ConversationContext')
            conv_context['inputs']['text'] = self.input_transcript
        else:
            conv_context: dict = {
                'inputs': {
                    "text": self.input_transcript,
                    "past_user_inputs": [],
                    "generated_responses": []
                },
                'history':initial_history
            }

        logger.debug(
            f"Req Session: {json.dumps(self.session_attributes, indent=4)} \n Type {type(self.session_attributes)}")

        logger.debug(
            f"Conversation Conext: {conv_context} \n Type {type(conv_context)}")

        # LLM
        langchain_bot = BedrockLangchainBot(
            prompt_template = prompt_template,
            # sm_endpoint_name = os.environ.get("ENDPOINT_NAME","cai-lex-hf-flan-bot-endpoint"),
            region_name = os.environ.get('AWS_REGION',"us-east-1"),
            lex_conv_history = json.dumps(conv_context['history'])
        )
        
        llm_response = langchain_bot.call_llm(user_input=self.input_transcript)
        curr_context = conv_context['inputs']
        self.text = llm_response

        curr_context["past_user_inputs"].append(self.input_transcript)
        curr_context["generated_responses"].append(self.text)
        conv_context['inputs']=curr_context
        conv_context['history'][CHAT_HISTORY] = conv_context['history'][CHAT_HISTORY] + self.input_transcript + f"\nAI: {self.text}" +"\nHuman: "

        self.intent_request['res']['session']['ConversationContext'] = conv_context

        self.intent_request['res']['message'] = self.text
        self.intent_request['res']['type'] = "plaintext"

        logger.debug(
            f"Response Generated: {json.dumps(self.intent_request['res'], indent=4)}")

        return self.intent_request
