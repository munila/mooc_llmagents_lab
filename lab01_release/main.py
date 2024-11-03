from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import math


def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    with open("restaurant-data.txt", "r") as file:
        all_reviews = file.readlines()
    restaurant_reviews = []
    for element in all_reviews:
        reviewed_restaurant, review = element.split(".", 1)
        reviewed_restaurant_letters = "".join(
            filter(str.isalpha, reviewed_restaurant.lower())
        )
        restaurant_name_letters = "".join(
            filter(str.isalpha, restaurant_name.lower())
        )
        if reviewed_restaurant_letters == restaurant_name_letters:
            restaurant_reviews.append(review)
    restaurant_data = {
        restaurant_name: restaurant_reviews
    }
    return restaurant_data


def calculate_overall_score(
        restaurant_name: str,
        food_scores: List[int],
        customer_service_scores: List[int],
) -> Dict[str, float]:
    joint_scores = []
    for i in range(len(food_scores)):
        joint_score = math.sqrt(food_scores[i] ** 2 * customer_service_scores[i]) \
                      * 1 / (len(food_scores) * math.sqrt(125))
        joint_scores.append(joint_score)
    final_score = sum(joint_scores) * 10
    return {
        restaurant_name: final_score
    }


# TODO: feel free to write as many additional functions as you'd like.
# Do not modify the signature of the "main" function.
def main(user_query: str):
    entrypoint_agent_system_message = "You are an helpful assistant, that can communicate with different agents " \
                                      "about restaurant reviews"
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    entrypoint_agent = ConversableAgent(
        "entrypoint_agent",
        system_message=entrypoint_agent_system_message,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    data_fetch_agent = ConversableAgent(
        name="data_fetch_agent",
        system_message="You are a helpful AI assistant. "
                       "You can help with generating function signatures to query restaurant reviews"
                       "Reply `TERMINATE` after the function is called",
        llm_config=llm_config,
        max_consecutive_auto_reply=1,
        human_input_mode="NEVER",
    )
    data_fetch_agent.register_for_llm(description="A function to fetch data of a restaurant")(fetch_restaurant_data)
    entrypoint_agent.register_for_execution()(fetch_restaurant_data)

    system_prompt_analyze_agent = """
    You are a helpful assistant which is responsible
    to generate scores based on restaurant reviews.
    You will get a list of reviews which is in between these backets [] 
    and each review is written in quotes.
    First enumerate all reviews which are given in the query. 
    For each review you should generate two scores,
    The score should be between 1 and 5 and for quantification
    one food_score and one customer_service_score .
    the following rules should be applied:
    Score 1/5 has one of these adjectives: awful, horrible, or disgusting.
    Score 2/5 has one of these adjectives: bad, unpleasant, or offensive. 
    Score 3/5 has one of these adjectives: average, uninspiring, or forgettable.
    Score 4/5 has one of these adjectives: good, enjoyable, or satisfying.
    Score 5/5 has one of these adjectives: awesome, incredible, or amazing.
    Please put your output into two lists e.g.:
    food_score: [2, 5, 2];
    customer_service_score: [3, 4, 3].
    In order to solve this task think set by step:
    1. Enumerate all reviews which are given in the query and look at them iteratively.
    2. Extract the keywords associated with food per review.
    3. Create a score for food
    4. Extract the keyword associated with customer service per review. 
    5. Create a score for customer service
    6. Put all food_scores into a list
    7. Put all customer_service_scores into a list
    """

    analyze_agent = ConversableAgent(
        name="analyze_agent",
        system_message=system_prompt_analyze_agent,
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    init_message_analyze = "According to the summary of the last conversations, " \
                                 "what are the scores of the restaurant"

    system_prompt_scoring_agent = """
    You are a helpful assistant. Your aim is to generate function calls which 
    calculate a score for a restaurant.
    Therefore, you take a look at the messages you get, extract two lists and 
    generate the correct function call to execute.
    """
    scoring_agent = ConversableAgent(
        name="scoring_agent",
        system_message=system_prompt_scoring_agent,
        llm_config=llm_config,
        max_consecutive_auto_reply=1,
        human_input_mode="NEVER",
    )
    init_message_final_score = f"""
    According to the summary, can you generate the function call
    to calculate the final score of {user_query}
    """
    scoring_fn_description = """
    A function to calculate a score for a restaurant based on a list of 
    food quality scores and service quality scores 
    """
    scoring_agent.register_for_llm(description=scoring_fn_description)(calculate_overall_score)
    entrypoint_agent.register_for_execution()(calculate_overall_score)

    result = entrypoint_agent.initiate_chats(
        [
            {
                "recipient": data_fetch_agent,
                "message": f"What are the review for {user_query}?",
            },
            {
                "recipient": analyze_agent,
                "message": init_message_analyze,
                "max_turns": 1,
            },
            {
                "recipient": scoring_agent,
                "message": init_message_final_score,
            },
        ]
    )
    print(result)


# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
