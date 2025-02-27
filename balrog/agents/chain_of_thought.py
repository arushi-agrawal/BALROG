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

        short_term = obs["text"]["short_term_context"]
        long_term = obs["text"].get("long_term_context", "")
        query = f"{short_term}".strip()

        logger.info(f"Generated query: {query}")
        # logger.info(f"Short Term query: {short_term.strip()}")
        # logger.info(f"Long Term query: {long_term.strip()}")

        retrieved_docs = self.retriever.search(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        logger.info(f"Retrieved docs 1st: {retrieved_docs[0]}")
        logger.info(f"Retrieved docs 2nd: {retrieved_docs[1]}")
        logger.info(f"Retrieved docs 3rd: {retrieved_docs[2]}")

        messages = self.prompt_builder.get_prompt()

        # Add CoT-specific instructions to the prompt
        cot_instructions = """
First think about what's the best course of action step by step.
Finally, provide a single output action at the end of the message in the form of: ACTION: <action>
        """.strip()

        rag_instructions = """
        Go through the retrieved documents and consider the information they provide. The documents are there to help you make an informed decision. They have information about game mechanics and optimal strategies. 
        """
        rag_instructions += "\n\n" + "Here are the retrieved documents:\n\n"
        for doc in retrieved_docs:
            rag_instructions += doc + "\n\n"
        rag_instructions.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + rag_instructions

        messages[-1].content += "\n\n" + cot_instructions

        # logger.info(f"Prompt messages: {messages}")

        # Generate the CoT reasoning
        cot_reasoning = self.client.generate(messages)

        # Extract the final answer from the CoT reasoning
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
