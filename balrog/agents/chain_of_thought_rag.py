import copy
import re

from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper
from balrog.prompt_builder.history import Message

from balrog.agents.agent_rag_utils import *
import logging

logger = logging.getLogger(__name__)


class ChainOfThoughtRAGAgent(BaseAgent):
    """An agent that performs actions using a chain-of-thought reasoning process."""

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, config):
        """Initialize the ChainOfThoughtRAGAgent with a client, prompt builder, and configuration.

        Args:
            client_factory (LLMClientWrapper): A factory for creating the LLM client instance.
            prompt_builder (PromptBuilder): Object to build prompts for the agent.
            config: Configuration object containing settings for the agent.
        """
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.retriever = NethackWikiSearch(config)
        self.retriever.load_index()
        self.remember_cot = config.agent.remember_cot

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

        short_term_context = obs["text"]["short_term_context"]
        long_term_context = obs["text"].get("long_term_context", "")
        context = f"{short_term_context} {long_term_context}".strip()
        # context = f"{short_term_context}".strip()
        logger.debug(f"Context: {context}")

        system_prompt = self.prompt_builder.system_prompt

        query_message = copy.deepcopy(messages)

        rag_query_prompt ="""
        Based on the game state above and the overall game instructions, generate a query that will help retrieve the most relevant strategic advice from the NetHack guide. 
        Your query could be about, but not limited to:
        - Key aspects of the current game state (e.g., inventory items, nearby threats, environmental features).
        - Whether you need offensive, defensive, or general guidance.
        - Specific details that will narrow down the retrieval to a useful topic.
        Your query must be a short phrase (maximum 8 words) that summarizes the primary strategic decision. Do not include multiple questions or detailed game state descriptions.
        For example:
        - "Uses for wand"
        - "Defeat dragon"
        Please output your query in the following format:
        Query: <query>
        """

        if messages and messages[-1].role == "user":
            query_message[-1].content += "\n\n" + rag_query_prompt

        rag_response = self.client.generate(query_message)

        rag_query = rag_response.completion
        rag_query = rag_query.split("Query:")[1].strip()

        logger.info(f"RAG query: {rag_query}")



        rag_docs = self.retriever.search(rag_query)
        rag_context = "\n".join([doc for doc in rag_docs])
        # logger.info(f"RAG context retrieved: {rag_context}")

        rag_context_summary = f"""Given the current state context and the retrieved RAG results, summarize the most relevant information for a NetHack player that they can 
        use to make a decision.
        - If you see a direction such as northnortheast, it means you should first move in the north direction and then the northeast direction. Give the
        direction in the order of the first direction and then the second direction.
        Example: context observation: gold piece near westsouthwest -> move west and then southwest
        Current State context:
        {context}
        RAG Results:
        {rag_context}
        Extract the most useful information from the retrieved RAG results. Be mindful of not omitting any information that specifies technical details that will be useful in making the next decision.
        Your final output should be in the following format, do not add anything else before or after the format. Output Format:
        Current State Summary: <summary in less than 50 words>
        Retrieved Summarized RAG Results:
        - <Most relevant information from the retrieved RAG result 1>
        - <Most relevant information from the retrieved RAG result 2>
        - <...so on for all retrieved RAG results...>
        """

        rag_summary = self.client.generate([Message(role="user", content=rag_context_summary)])
        rag_summary = rag_summary.completion

        # logger.info(f"RAG summary: {rag_summary}")
        messages = self.prompt_builder.get_prompt()

        rag_usage_prompt =f"""
            Below is the retrieved context from the RAG database. Use this information to help you make a decision.
            {rag_summary}
            """

        messages[-1].content += "\n\n" + rag_usage_prompt


        # Add CoT-specific instructions to the prompt
        cot_instructions = """
First think about what's the best course of action step by step.
Then, you must choose exactly one of the listed actions and output a single action at the end of the message in the form of: ACTION: <action>
Explain your action choice in not more than 20 words.
        """.strip()

        messages[-1].content += "\n\n" + cot_instructions

        # Generate the CoT reasoning
        cot_reasoning = self.client.generate(messages)

        # Extract the final answer from the CoT reasoning
        final_answer = self._extract_final_answer(cot_reasoning)
        # logger.info(f"Final answer: {final_answer}")

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