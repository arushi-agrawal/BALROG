import copy
import re
import logging

from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper
from balrog.agents.agent_rag_utils import *

logger = logging.getLogger(__name__)


class ChainOfThoughtAgent(BaseAgent):
    """An agent that performs actions using a chain-of-thought reasoning process."""

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, config):
        """Initialize the ChainOfThoughtAgent with a client, prompt builder, and configuration.

        Args:
            client_factory (LLMClientWrapper): A factory for creating the LLM client instance.
            prompt_builder (PromptBuilder): Object to build prompts for the agent.
            config: Configuration object containing settings for the agent.
        """
        super().__init__(client_factory, prompt_builder)
        self.remember_cot = config.agent.remember_cot
        self.retriever = NethackWikiSearch(config)
        self.retriever.load_index()

    def act(self, obs, prev_action=None):
        """Generate the next action using chain-of-thought reasoning based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the final selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        query_instructions = """
        Asses the current situation properly. Now imagine that you have a information rich document for NetHac containing all the things that you encounter in game. 
        Output a concise 4-5 words sentence of what you would like to get from the document. For example: "fountain", or "defeat a fox?" Reply in the form of: QUESTION: <question>
        """.strip()

        query_message = copy.deepcopy(messages)
        if messages and messages[-1].role == "user":
            query_message[-1].content += "\n\n" + query_instructions
        # messages[-1].content += "\n\n" + query_instructions

        # logger.info(f"Message after asking for query: {query_message}")

        query_reasoning = self.client.generate(query_message)
        query = self._extract_question(query_reasoning)
        
        # messages[-1].content.replace(query_instructions, "")
        # logger.info(f"Message after removing query instructions: {messages}")
        question = query.reasoning.split("QUESTION:")[-1].strip()
        logger.info(f"Extracted question: {question}")

        retrieved_docs = self.retriever.search(question)
        # logger.info(f"Retrieved {len(retrieved_docs)} documents")
        # logger.info(f"Retrieved docs 1st: {retrieved_docs[0][:100]}")
        # logger.info(f"Retrieved docs 2nd: {retrieved_docs[1][:100]}")
        # logger.info(f"Retrieved docs 3rd: {retrieved_docs[2][:100]}")


        rag_instructions = """
        Now go through the retrieved documents and consider the information they provide. The documents are there to help you make an informed decision.
        """
        rag_instructions += "\n\n" + "Here are the retrieved documents:\n\n"
        for doc in retrieved_docs:
            rag_instructions += doc + "\n\n"
        rag_instructions.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + rag_instructions


        # Add CoT-specific instructions to the prompt
        cot_instructions = """
Now think about what's the best course of action step by step. The retrieved documents might not be completely accurate, so use your best judgement.
Finally, provide a valid single output action (**crosscheck that the output action is a valid action given in the list of actions**) at the end of the message in the form of: ACTION: <action>
        """.strip()

        messages[-1].content += "\n\n" + cot_instructions

        # Generate the CoT reasoning
        cot_reasoning = self.client.generate(messages)
        # logger.info(f"Complete message: {messages}")
        
        # # Extract the final answer from the CoT reasoning
        final_answer = self._extract_final_answer(cot_reasoning)
        return final_answer

    def _extract_final_answer(self, reasoning):
        """Extract the final action from the chain-of-thought reasoning response.

        Args:
            reasoning (LLMResponse): The response containing CoT reasoning and action.

        Returns:
            LLMResponse: The response with the extracted final action.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        answer = copy.deepcopy(reasoning)
        self.prompt_builder.update_reasoning(reasoning.completion)
        answer = answer._replace(reasoning=answer.completion)
        answer = answer._replace(completion=filter_letters(answer.completion).split("ACTION:")[-1].strip())

        return answer
    
    def _extract_question(self, reasoning):
        """Extract the question from the chain-of-thought reasoning response.

        Args:
            reasoning (LLMResponse): The response containing CoT reasoning and action.

        Returns:
            str: The question extracted from the reasoning.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        question = copy.deepcopy(reasoning)
        # self.prompt_builder.update_reasoning(reasoning.completion)
        question = question._replace(reasoning=question.completion)
        question = question._replace(completion=filter_letters(question.completion).split("QUESTION:")[-1].strip())

        return question
