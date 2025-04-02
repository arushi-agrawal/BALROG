import copy
import re
import logging
from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper
# from balrog.agents.utils.rag import RAG
from balrog.prompt_builder.history import Message
# from balrog.environments.nle.base import NLELanguageWrapper
from balrog.agents.agent_rag_utils import *

import time
import psutil
import gc

logger = logging.getLogger(__name__)

# all_nle_action_map = NLELanguageWrapper.all_nle_action_map

# available_actions = [
#                 action_strs[0]
#                 for action, action_strs in all_nle_action_map.items()
#                 if action in USEFUL_ACTIONS
#             ]
# single_chars = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [
#                 chr(i) for i in range(ord("A"), ord("Z") + 1)
#             ]
# single_digits = [str(i) for i in range(10)]
# double_digits = [f"{i:02d}" for i in range(100)]
# yes_no = ["yn", 'n']
# all_actions = available_actions + single_chars + single_digits + double_digits + yes_no
# all_actions_str = "\n-".join(all_actions)

ACTIONS = {
    "north": "move north",
    "east": "move east",
    "south": "move south",
    "west": "move west",
    "northeast": "move northeast",
    "southeast": "move southeast",
    "southwest": "move southwest",
    "northwest": "move northwest",
    "far north": "move far north",
    "far east": "move far east",
    "far south": "move far south",
    "far west": "move far west",
    "far northeast": "move far northeast",
    "far southeast": "move far southeast",
    "far southwest": "move far southwest",
    "far northwest": "move far northwest",
    "up": "go up a staircase",
    "down": "go down a staircase (tip: you can only go down if you are standing on the stairs)",
    "wait": "rest one move while doing nothing",
    "more": "display more of the message (tip: ONLY ever use when current message ends with --More--)",
    "annotate": "leave a note about the level",
    "apply": "apply (use) a tool",
    "call": "name a monster or object, or add an annotation",
    "cast": "cast a spell",
    "close": "close an adjacent door",
    "open": "open an adjacent door",
    "dip": "dip an object into something",
    "drop": "drop an item",
    "droptype": "drop specific item types (specify in the next prompt)",
    "eat": "eat something (tip: replenish food when hungry)",
    "esc": "exit menu or message",
    "engrave": "engrave writing on the floor (tip: Elbereth)",
    "enhance": "advance or check weapons skills",
    "fire": "fire ammunition from quiver",
    "fight": "fight a monster (even if you only guess one is there)",
    "force": "force a lock",
    "inventory": "show your inventory",
    "invoke": "invoke ",
    "jump": "jump to a location",
    "kick": "kick an enemy or a locked door or chest",
    "look": "look at what is under you",
    "loot": "loot a box on the floor",
    "monster": "use a monster's special ability (when polymorphed)",
    "offer": "offer a sacrifice to the gods (tip: on an aligned altar)",
    "overview": "display an overview of the dungeon",
    "pay": "pay your shopping bill",
    "pickup": "pick up things at the current location",
    "pray": "pray to the gods for help",
    "puton": "put on an accessory",
    "quaff": "quaff (drink) something",
    "quiver": "select ammunition for quiver",
    "read": "read a scroll or spellbook",
    "remove": "remove an accessory",
    "rub": "rub a lamp or a stone",
    "search": "search for hidden doors and passages",
    "swap": "swap wielded and secondary weapons",
    "takeoff": "take off one piece of armor",
    "takeoffall": "take off all armor",
    "teleport": "teleport to another level (if you have the ability)",
    "throw": "throw something (e.g. a dagger or dart)",
    "travel": "travel to a specific location on the map (tip: in the next action, specify > or < for stairs, { for fountain, and _ for altar)",
    "twoweapon": "toggle two-weapon combat",
    "untrap": "untrap something",
    "wear": "wear a piece of armor",
    "wield": "wield a weapon",
    "wipe": "wipe off your face",
    "zap": "zap a wand",
    "minus": "-",
    "space": " ",
    "apos": "'",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}

action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())

refined_system_prompt = f"""
You are an agent playing NetHack. Below is a set of allowed actions followed by a short description of each action.
For example:
north: move north
east: move east
south: move south
west: move west
northeast: move northeast

Here is the list of all allowed actions:

{action_strings}.

Tips:
- Taking the stairs up on level 1 without Amulet of Yendor will quit the game and you will lose. Do not take the stairs up on level 1 without the Amulet of Yendor.
- When the message asks for a completion, such as: "What do you want to eat? [d or ?*]", you should respond with a single character corresponding to the item you want to eat/use.
    - For example, "What do you want to eat? [dgh or ?*]" -> Possible answers are "d", "g", or "h" to eat the associated food.
- When the message asks for a direction, such as: "In what direction?" you should respond with a direction.
- When the message has --More-- at the end, your next action should be "more" to see the rest of the message.
- Explore the environment to find the stairs down to the next level.
- Always carefully read the last message to understand the current state of the game and decide your next action accordingly.
- If you keep moving in the same direction, you will eventually hit a wall and stop moving. Your message might be: "It's solid stone", or "It's a wall". Change your action to move in another direction to continue exploring the environment.
- Read the language observation carefully and look at ascii map or image observation provided to decide the next action to take and where to move next.
- You can attack monsters by moving into them.

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible in the game. Explore all possible areas. DO NOT FIXATE ON ONE ACTION/ STRATEGY FOR TOO LONG. KEEP THE GAME MOVING.

""".strip()


class RobustCoTRAGAgent(BaseAgent):
    """An agent that performs actions using chain-of-thought reasoning with RAG-enabled retrieval."""

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, config):
        """Initialize the RobustCoTRAGAgent with a client, prompt builder, RAG instance, and configuration.

        Args:
            client_factory (LLMClientWrapper): A factory for creating the LLM client instance.
            prompt_builder (PromptBuilder): Object to build prompts for the agent.
            rag_instance: The RAG instance for retrieving relevant documents.
            config: Configuration object containing settings for the agent.
        """
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.retriever = NethackWikiSearch(config)
        self.retriever.load_index()
        self.remember_cot = config.agent.remember_cot
        logger.info("RobustCoTRAGAgent initialized")

    def act(self, obs, prev_action=None):
        """Generate the next action using chain-of-thought reasoning with RAG retrieval.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the final selected action.
        """
    
        start_time = time.time()
        # total_input_tokens = 0
        # total_output_tokens = 0
        
        try:
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
            - "Effective defensive tactics, limited weapons, staircase"
            - "Best potion usage, goblins, early game"
            - "Defeating a dragon"

            Please output your query in the following format:
            Query: <query>
            """

            if messages and messages[-1].role == "user":
                query_message[-1].content += "\n\n" + rag_query_prompt

            # logger.info(f"Query message to LLM for RAG: {query_message}")

            # Track RAG query tokens
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

            # logger.info(f"RAG Query: {rag_query}")
            # logger.info(f"RAG context: {rag_context}")
            # logger.info(f"RAG summary: {rag_summary}")

            messages = self.prompt_builder.get_prompt()

            rag_usage_prompt =f"""
Below is the retrieved context from the RAG database. Use this information to help you make a decision.
{rag_summary}
            """

            messages[-1].content += "\n\n" + rag_usage_prompt

            
            cot_instructions = """First, think about the best course of action.
Then, you must choose exactly one of the listed actions and output it strictly in the following format:

<|ACTION|>YOUR_CHOSEN_ACTION<|END|>

Explain your action choice in not more than 15 words.
"""

            messages[-1].content += "\n\n" + cot_instructions
            # logger.info(f"Final Prompt: {messages}")

            cot_reasoning = self.client.generate(messages)
            
            # total_input_tokens += cot_reasoning.input_tokens
            # total_output_tokens += cot_reasoning.output_tokens
            # logger.info(f"COT reasoning: {cot_reasoning.completion}")

            final_answer = self._extract_final_answer(cot_reasoning)

            end_time = time.time()
            # memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            # logger.info(f"""
            # Performance metrics:
            # - Time taken: {end_time - start_time:.2f}s
            # """)

            gc.collect()

            return final_answer
            

        except Exception as e:
            logger.error(f"Error in act(): {str(e)}", exc_info=True)
            # Return a safe default response in case of error
            return self.client.generate([Message(role="user", content="Output a single valid action in the format <|ACTION|>action<|END|>.")])

    def _extract_final_answer(self, reasoning):
        """Extract the final action from the chain-of-thought reasoning response.

        Args:
            reasoning (LLMResponse): The response containing CoT reasoning and action.

        Returns:
            LLMResponse: The response with the extracted final action in `completion`
                         and the entire chain-of-thought in `reasoning`.
        """
        try:
            final_answer = copy.deepcopy(reasoning)

            final_answer = final_answer._replace(reasoning=reasoning.completion)

            completion_text = reasoning.completion
            match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
            if match:
                extracted_action = match.group(1).strip()
            else:
                logger.warning("Failed to extract action using the strict format")
                extracted_action = self._fallback_extraction(completion_text)

            # Replace the final `completion` with only the extracted action
            final_answer = final_answer._replace(completion=extracted_action)

            return final_answer
            
        except Exception as e:
            logger.error(f"Error in _extract_final_answer(): {str(e)}")
            return reasoning

    def _fallback_extraction(self, text):
        """Fallback method to extract an action when the strict format fails."""
        # Filter to keep only alphabetic characters as a last resort
        return re.sub(r"[^a-zA-Z\s:]", "", text).strip()
    
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
    