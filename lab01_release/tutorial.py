from typing import Dict, List
from autogen import ConversableAgent
import sys
import os


def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    # TODO
    # This function takes in a restaurant name and returns the reviews for that restaurant.
    # The output should be a dictionary with the key being the restaurant name and the value being a list of reviews for that restaurant.
    # The "data fetch agent" should have access to this function signature, and it should be able to suggest this as a function call.
    # Example:
    # > fetch_restaurant_data("Applebee's")
    # {"Applebee's": ["The food at Applebee's was average, with nothing particularly standing out.", ...]}#

    with open("restaurant-data.txt", "r") as file:
        all_reviews = file.readlines()
    restaurant_reviews = []
    for element in all_reviews:
        reviewed_restaurant, review = element.split(".", 1)
        if reviewed_restaurant.lower() == restaurant_name.lower():
            restaurant_reviews.append(review)
    restaurant_data = {
        restaurant_name: restaurant_reviews
    }
    return restaurant_data


def main(query):
    # Let's first define the assistant agent that suggests tool calls.
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]}
    entrypoint_agent = ConversableAgent(
        name="Assistant",
        system_message="You are a helpful AI assistant. "
                       "You can help with fetching restaurant data. "
                       "Return 'TERMINATE' when the task is done.",
        llm_config=llm_config,
    )

    # The user proxy agent is used for interacting with the assistant agent
    # and executes tool calls.
    data_fetch_agent = ConversableAgent(
        name="Fetching",
        system_message="You are a helpful AI assistant. "
                       "You can help with fetching restaurant data. "
                       "Return 'TERMINATE' when the task is done.",
        llm_config=llm_config,
    )

    # Register the tool signature with the assistant agent.
    entrypoint_agent.register_for_llm(description="A function to fetch data of a restaurant")(fetch_restaurant_data)

    # Register the tool function with the user proxy agent.
    data_fetch_agent.register_for_execution()(fetch_restaurant_data)

    chat_result = data_fetch_agent.initiate_chat(entrypoint_agent, message=f"What is the review of {query}?", max_turns=3)
    print(chat_result)


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])