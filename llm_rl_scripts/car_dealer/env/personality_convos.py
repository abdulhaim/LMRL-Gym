import os
import openai
import random
import re
import json
import pickle as pkl

key = "OPEN_AI_KEY"
openai.api_key = key

def get_conversation_prompt(background_buyer, background_seller, budget, brand, classification, feature):
    # Directly get a conversation from the model
    intro_1 = "Simulate a conversation between an agent at a car dealership and a buyer. "
    intro_2 = "The agent wants to want to maximize the sell price of the car."
    buyer_preference = f"The buyer prefers {brand} {classification}, ideally also has the features of {feature}."
    buyer_budget = f"The buyer has a budget of ${budget}, but the agent does not know the budget beforehand. The buyer prefers not to pay more than the budget, but can be convinced to change their mind. "
    reward_type = "Whenever the buyer finishes a sentence, output the probability that the buyer will buy a car, and the purchase price. "
    output_format_1 = "Please output the predicted probability and price in the same line. "
    output_format_2 = "Please output the probability that the buyer will buy a car and the purchase price at the end of the conversation. "
    prompt = [
        {"role": "system", "content": ""},
        {"role": "user", "content": intro_1 + intro_2 + background_seller + background_buyer + buyer_preference + buyer_budget + reward_type + output_format_1 + output_format_2},
    ]

    return prompt

def get_purchase_prob(strings):
    # change things to lower case
    strings = strings.lower()
    # check whether the string contains the purchase probability, if so, return the probability
    if "probability" in strings or "chance" in strings:
        # check percentage matching
        match_float = re.search(r'\d+(\.\d+)?%', strings)
        match_float_decimal = re.search(r'\d+(\.\d+)?', strings)
        if match_float:
            return float(match_float.group(0)[:-1])/100 # removes the '%' symbol
        elif match_float_decimal:
            return float(match_float_decimal.group(0)) # directly return the float
        else:
            return 0
    else:
        return 0

def get_purchase_price(strings):
    # change things to lower case
    strings = strings.lower()
    # check whether the string contains the purchase price, if so, return the price
    if "purchase" in strings:
        # check percentage matching, after the string "purchase"
        purchase_index = strings.index("purchase")
        purchase_amount_str = strings[purchase_index+len("purchase"):]
        purchase_amount_str = strings[purchase_index+len("purchase"):]
        match_float_k = re.search(r'\$\d+(\.\d+)?k', strings)
        match_float_comma = re.search(r"\$([\d,]+(\.\d+)?)", strings)
        if match_float_k:
            first_index = match_float_k.group(0).find("$")
            return float(match_float_k.group(0)[1:-1])*1000 # removes the '$' symbol and 'k' symbol
        elif match_float_comma:
            first_index = match_float_comma.group(1).find("$")
            return float(match_float_comma.group(1).replace(",", "")) # removes the '$' symbol and ',' symbol
        else:
            return 0
    elif "price" in strings:
        # check percentage matching, after the string "price"
        purchase_index = strings.index("price")
        purchase_amount_str = strings[purchase_index+len("price"):]
        purchase_amount_str = strings[purchase_index+len("price"):]
        match_float_k = re.search(r'\$\d+(\.\d+)?k', strings)
        match_float_comma = re.search(r"\$([\d,]+(\.\d+)?)", strings)
        if match_float_k:
            first_index = match_float_k.group(0).find("$")
            return float(match_float_k.group(0)[1:-1])*1000 # removes the '$' symbol and 'k' symbol
        elif match_float_comma:
            first_index = match_float_comma.group(1).find("$")
            return float(match_float_comma.group(1).replace(",", "")) # removes the '$' symbol and ',' symbol
        else:
            return 0
    else:
        return 0


background_buyers = ["The buyer will never buy from the seller unless given a discount.", "The buyer will only buy the car if it meets all of their specifications.", "The buyer is extremely impatient and strongly dislikes unnecessary information about discounts and features."]
background_sellers = ["The agent never talks about the features of the car and loves to give discounts.", "The agent never gives discounts and loves to talk a lot about features of the car.", "The agent never gives discounts, never talks about the features of the car, and is very concise."]

brands = ["BMW", "Lexus", "Honda", "Toyota", "Mazda", "Audi", "Hyundai", "Porsche", "Tesla", "Volkswagen", "Ford", "Mercedes-Benz", "Subaru", "Porsche"]
features = ["Leather seats", "Sunroof/moonroof", "Heated seats", "Backup camera", "Navigation system", "Bluetooth", "Remote start", "Blind spot monitoring", "Third-row seating", "Apple CarPlay/Android Auto"]
budgets = ["10k", "30k", "50k", "70k", "90k"]


# Classifications source: https://www.jdpower.com/cars/body-styles
classifications = ["SUV", "Crossover", "Sedan", "Truck", "Hatchback", "Convertible", "Luxury", "Coupe", "Hybrid", "Electric", "Van", "Minivan", "Sports"]
classification = random.choice(classifications)

seed = 1 
random.seed(seed)
count_convos = 1
verbose = False 

# Set total number of conversations
conversation_num = 50
conversations = []
for k in range(len(background_buyers)):
    for j in range(len(background_sellers)):
        background_buyer = background_buyers[k]
        background_seller = background_sellers[j]

        for conv in range(conversation_num):
            # Randomly select personality, brand, budget, and features
            print("conversation", count_convos)
            count_convos += 1
            background_buyer = random.choice(background_buyers)
            background_seller = random.choice(background_sellers)

            brand = random.choice(brands)
            num_features = random.randint(1, len(features))
            selected_feature = random.sample(features, k=num_features)
            feature = ', '.join(selected_feature)
            budget = random.choice(budgets)

            if verbose: 
                print(background_buyer)
                print(background_seller)
                print(brand)
                print(budget)
                print(classification)
                print(feature)

            try: 
                conversation = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=get_conversation_prompt(background_buyer, background_seller, budget, brand, classification, feature),
                    temperature=0.7,
                    max_tokens=2048,
                )

            except (openai.APIError, openai.OpenAIError) as e:
                print(e)
                print("sleeping for 10 seconds.")
                time.sleep(10)
                return None

            time.sleep(0.5)

            conversation = conversation.choices[0]['message']['content'].strip()
            if conversation[0] == "\"":
                conversation = conversation[1:]
            if conversation[-1] == "\"":
                conversation = conversation[:-1]

            one_conv = ["Agent: " + line.strip() for line in conversation.split("Agent: ") if len(line.strip()) > 0]
            if verbose: 
                print(conversation)
                print("********")

            lines = []

            for i in range(len(one_conv)):
                purchase_prob = get_purchase_prob(one_conv[i])
                purchase_price = get_purchase_price(one_conv[i])
                stripped_string = [line.strip() for line in one_conv[i].split("\n") if len(line.strip()) > 0]
                for j in range(len(stripped_string)):
                    if stripped_string[j].startswith('Agent: '):
                        _, dealer_text = stripped_string[j].split('Agent: ')
                        lines.append({"role": "Dealer", "text": dealer_text, "purchase_prob": purchase_prob, "purchase_price": purchase_price})
                    elif stripped_string[j].startswith('Buyer: '):
                        _, buyer_text = stripped_string[j].split('Buyer: ')
                        lines.append({"role": "Buyer", "text": buyer_text, "purchase_prob": purchase_prob, "purchase_price": purchase_price})
            if verbose: 
                print(lines)
                print("========")

            conversation_data = {
                "buyer_info": {
                    "background_buyer": background_buyer,
                    "background_seller": background_seller,
                    "preferred_brands": brand,
                    "preferred_features": selected_feature,
                    "budget": budget,
                },
                "original_conversation": conversation,
                "lines": lines,
            }
            conversations.append(conversation_data)

        # Save file into json
        with open("personality_conversations/personality_convos_seed_" + str(seed) + "json", "w") as f:
            json.dump(conversations, f, indent=4)
