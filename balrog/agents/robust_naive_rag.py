import copy
import re

from balrog.agents.base import BaseAgent

from balrog.prompt_builder.history import Message

from balrog.agents.agent_rag_utils import *
import logging

logger = logging.getLogger(__name__)


class RobustNaiveRAGAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, client_factory, prompt_builder, config):
        """Initialize the NaiveAgent with a client and prompt builder."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.retriever = NethackWikiSearch(config)
        self.retriever.load_index()

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
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
        - "Defensive tactics, boulder, door"
        - "Best potion usage, goblins"
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
        Current State Summary:<summary in less than 30 words>
        Retrieved Summarized RAG Results:
        - <Most relevant information from the retrieved RAG result 1>
        - <Most relevant information from the retrieved RAG result 2>
        - <...so on for all retrieved RAG results...>
        """

        rag_summary = self.client.generate([Message(role="user", content=rag_context_summary)])
        rag_summary = rag_summary.completion

        messages = self.prompt_builder.get_prompt()

        rag_usage_prompt =f"""
            Below is the retrieved context from the RAG database. Use this information to help you make a decision.
            {rag_summary}
            """

        messages[-1].content += "\n\n" + rag_usage_prompt

        # Updated instructions to require a very strict output format
        naive_instruction = """
You must choose exactly one of the listed actions and output it strictly in the following format:

<|ACTION|>YOUR_CHOSEN_ACTION<|END|>

Replace YOUR_CHOSEN_ACTION with the chosen action. Output no other text, explanation, or reasoning.
""".strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)
        final_answer = self._extract_final_answer(response)
        return final_answer

    def _extract_final_answer(self, answer):
        """Extract the action from the completion by looking for <|ACTION|> and <|END|> tags.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response containing just the extracted action.
        """
        completion_text = answer.completion
        # Use a regex to find the text inside <|ACTION|> and <|END|>
        match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
        if match:
            extracted_action = match.group(1).strip()
        else:
            # If no match is found, fallback to the original completion or handle as needed
            extracted_action = completion_text.strip()

        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(completion=extracted_action)

        return final_answer
