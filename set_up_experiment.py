import numpy as np
from tqdm import tqdm # for progress bar
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time
from textwrap import dedent
from tqdm import tqdm
from google.colab import files
from google.colab import userdata

################################################################################
########## Utilities for Text Processing and LLM-Response Parsing ##############
################################################################################
class_to_id_map = {
  "apple_pie": 0,
  "baby_back_ribs": 1,
  "baklava": 2,
  "beef_carpaccio": 3,
  "beef_tartare": 4,
  "beet_salad": 5,
  "beignets": 6,
  "bibimbap": 7,
  "bread_pudding": 8,
  "breakfast_burrito": 9,
  "bruschetta": 10,
  "caesar_salad": 11,
  "cannoli": 12,
  "caprese_salad": 13,
  "carrot_cake": 14,
  "ceviche": 15,
  "cheesecake": 16,
  "cheese_plate": 17,
  "chicken_curry": 18,
  "chicken_quesadilla": 19,
  "chicken_wings": 20,
  "chocolate_cake": 21,
  "chocolate_mousse": 22,
  "churros": 23,
  "clam_chowder": 24,
  "club_sandwich": 25,
  "crab_cakes": 26,
  "creme_brulee": 27,
  "croque_madame": 28,
  "cup_cakes": 29,
  "deviled_eggs": 30,
  "donuts": 31,
  "dumplings": 32,
  "edamame": 33,
  "eggs_benedict": 34,
  "escargots": 35,
  "falafel": 36,
  "filet_mignon": 37,
  "fish_and_chips": 38,
  "foie_gras": 39,
  "french_fries": 40,
  "french_onion_soup": 41,
  "french_toast": 42,
  "fried_calamari": 43,
  "fried_rice": 44,
  "frozen_yogurt": 45,
  "garlic_bread": 46,
  "gnocchi": 47,
  "greek_salad": 48,
  "grilled_cheese_sandwich": 49,
  "grilled_salmon": 50,
  "guacamole": 51,
  "gyoza": 52,
  "hamburger": 53,
  "hot_and_sour_soup": 54,
  "hot_dog": 55,
  "huevos_rancheros": 56,
  "hummus": 57,
  "ice_cream": 58,
  "lasagna": 59,
  "lobster_bisque": 60,
  "lobster_roll_sandwich": 61,
  "macaroni_and_cheese": 62,
  "macarons": 63,
  "miso_soup": 64,
  "mussels": 65,
  "nachos": 66,
  "omelette": 67,
  "onion_rings": 68,
  "oysters": 69,
  "pad_thai": 70,
  "paella": 71,
  "pancakes": 72,
  "panna_cotta": 73,
  "peking_duck": 74,
  "pho": 75,
  "pizza": 76,
  "pork_chop": 77,
  "poutine": 78,
  "prime_rib": 79,
  "pulled_pork_sandwich": 80,
  "ramen": 81,
  "ravioli": 82,
  "red_velvet_cake": 83,
  "risotto": 84,
  "samosa": 85,
  "sashimi": 86,
  "scallops": 87,
  "seaweed_salad": 88,
  "shrimp_and_grits": 89,
  "spaghetti_bolognese": 90,
  "spaghetti_carbonara": 91,
  "spring_rolls": 92,
  "steak": 93,
  "strawberry_shortcake": 94,
  "sushi": 95,
  "tacos": 96,
  "takoyaki": 97,
  "tiramisu": 98,
  "tuna_tartare": 99,
  "waffles": 100
}

id_to_class_map = { id : cls for cls, id in class_to_id_map.items()}

classes = [key for key in class_to_id_map.keys()]

def parse_attributes_string(response):
    attributes = []
    for line in response.splitlines():
        # remove dashes
        line = line[2:]

        attributes.append(line)

    return attributes

def parse_response(response):
    lines = []
    for line in response.splitlines():
        # remove dashes
        line = line[2:]

        lines.append(line)

    return lines

# What the LLM told us are valid
def get_whitelist(response):
    """
    Functionally, "handles" duplicates
    """
    K = len(classes)
    whitelist = np.zeros(K)

    # Process response
    response = parse_response(response)

    for line in response:
        line = line.lower().split()
        line = "_".join(line)

        if line in classes:
            idx = class_to_id_map[line]
            whitelist[idx] = 1

    return whitelist
#===============================================================================


def gen_preferences_prompt():
    """
    """
    template = dedent(f"""\
            Q: What are some standard food preferences someone might have?
            A: There are several common food preferences a person could have. They may want to:
            - be vegetarian
            - avoid nuts
            - keep kosher
            - eat vegan
            - avoid gluten
            - be pescetarian
            - keep away from dairy

            Q: What are some creative food preferences someone might have?
            A: There are several creative food preferences a person could have. They may want to:
            - get mediterranean food
            - eat tart flavors
            - have something that smells strong
            - eat umami flavors
            - eat tomato-based dishes
            - find summer dishes
            - eat something stinky
            - eat carribean flavors
            - have pirate-themed cuisine
            - eat surprising flavors
            - have exotic flavors
            - be adventurous
            - diet
            - lose weight
            - be health-conscious
            - eat food that goes with kimchi
            - eat something that smells like a fire
            - try something floral
            - order from the kids menu
            - have acidic food
            - have a smelly meal

            Q: What are some interesting food preferences someone might have?
            A: There are several interesting food preferences a person could have. They may want to:
            -"""
    )

    return template

# The below commented-out prompt function/template did not work well in initial testing
# on the development set (a subset of the Food101 trainset); the one below
# seemed to work better.
# def gen_descriptors_prompt(preference):
#     """
#     Notes on demonstrations in prompt template:
#     - The first question-answer pair ("lemur" topic) comes from
#     "Visual Classification via Description from Large Language Models",
#     Sachit Menon, Carl Vondrick: https://arxiv.org/abs/2210.07183

#     - The second question-answer pair ("tornado" topic) was generated by Mistral
#     spontaneously after answering a target question. Since it seemed spot on,
#     I decided to include it in future prompts as an exemplar.
#     """

#     template = dedent(f"""\
#             Q: What are useful features for distinguishing a lemur in a photo?
#             A: There are several useful visual features to tell there is a lemur in a photo:
#             - four-limbed primate
#             - black, grey, white, brown, or red-brown
#             - wet and hairless nose with curved nostrils
#             - long tail
#             - large eyes
#             - furry bodies
#             - clawed hands and feet

#             Q: What are useful features for distinguishing a tornado in a photo?
#             A: There are several useful visual features to tell there is a tornado in a photo:
#             - dark, rotating cloud
#             - funnel cloud
#             - debris in the air
#             - damage to buildings
#             - damage to trees
#             - damage to cars
#             - damage to power lines
#             - damage to roads

#             Q: What are useful features for distinguishing healthy food in a photo?
#             A: There are several useful visual features to tell there is healthy food in a photo:
#             - seasonal food
#             - green vegetables
#             - fresh food
#             - natural food
#             - simple food
#             - unprocessed food
#             - organic food
#             - whole food

#             Q: What are useful features for distinguishing food in a photo that matches a preference to {preference}?
#             A: There are several useful visual features to tell food in a photo matches a preference to {preference}:
#             -"""
#     )

#     return template

def gen_descriptors_prompt(preference):
    """
    """
    template = dedent(f"""\
            Q: What kind of foods do I want if I want to **eat sugary food**? Answer as a very simple, short list.
            A: You may want the following types of foods:
            - Desserts
            - Candies and chocolates
            - Sugary drinks
            - Pastries & baked goods

            Q: What kind of foods do I want if I want to **be kosher**? Answer as a very simple, short list.
            A: You may want the following types of foods:
            - Kosher meat
            - Permitted fish
            - Fresh produce
            - Certified processed foods

            Q: What kind of foods do I want if I want to **have noodles**? Answer as a very simple, short list.
            A: You may want the following types of foods:
            - Asian noodles
            - Italian pastas
            - Soupy noodles
            - Quick cup noodles

            Q: What kind of foods do I want if I want to **make something on a porch in the summer**? Answer as a very simple, short list.
            A: You may want the following types of foods:
            - Grilled fare
            - Cool salads
            - Summery drinks
            - Easy finger foods
            - Frozen treats

            Q: What kind of foods do I want if I want to **{preference}**? Answer as a very simple, short list.
            A: You may want the following types of foods:
            -"""
    )
    return template

def generate_prompt(preference=None, mode=None):
    """
    """
    prompt = None

    # if mode == "relevant_classes":
    #     prompt = gen_relevant_classes_prompt(preference)

    if mode == "descriptors":
        prompt = gen_descriptors_prompt(preference)

    if mode == "preferences":
        prompt = gen_preferences_prompt()

    if not mode:
        print("Invalid mode provided.")

    return prompt

def extract_target_list(response_with_prepended_hyphen):
    """
    Trims any additional content (rambling) after the target list is produced
    """
    lines = []
    for line in response_with_prepended_hyphen.splitlines():
        if len(line) > 2:
            if line[:2] != "- ":
                # line is not bulleted, indicating list ended above it
                break

            # remove dashes
            line = line[2:]
            lines.append(line)

        else:
            # line is too short, indicating list ended above it
            break

    return lines

class LLM():
    def __init__(self, model_id="mistralai/Mistral-7B-v0.1"):
        """
        Load the model in half-precision to make it fit in Colab runtime with T4
        GPU. Still, must be set to high-memory!

        Sources for skeleton code to load and use the model in half-precision
        - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
        - https://huggingface.co/mistralai/Mixtral-8x7B-v0.1#in-half-precision
        """
        self.device = "cuda" # the device to load the model onto

        self.model = AutoModelForCausalLM.from_pretrained(model_id,
                                                          torch_dtype=torch.float16,
                                                          token=userdata.get('HF')).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=userdata.get('HF'))

    def get_response(self, preference, mode, num_iters=1):
        """
        """
        # if mode == "relevant_classes":
        #     text = generate_prompt(preference, mode)

        if mode == "descriptors":
            text = generate_prompt(preference, mode)
            token_limit = 64

        if mode == "preferences":
            text = generate_prompt(preference=None, mode=mode)
            token_limit = 250

        extracted = []
        for i in range(num_iters):

            encodeds = self.tokenizer([text], return_tensors="pt")
            model_inputs = encodeds.to(self.device)

            generated_ids = self.model.generate(**model_inputs, max_new_tokens=token_limit, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)

            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            response_only = decoded[ len(text): ]

            response_plus_open_hyphen = "-" + response_only

            extracted_list = extract_target_list(response_plus_open_hyphen)

            extracted += extracted_list # concatenate with running list

        return extracted

    def get_all_responses(self, preferences=None, mode=None, colab_download=False):
        """
        Returns:

            - a dictionary of preference: List[str] pairs if mode ==
              "relevant_classes" or "descriptors"

            - a list of strings if mode == "preferences"
        """
        responses = {}

        # if mode == "relevant_classes":
        #     for preference in preferences:
        #         response = self.get_response(preference, mode, num_iters=2)
        #         responses[preference] = response

        if mode == "descriptors":
            for preference in tqdm(preferences):
                response = self.get_response(preference, mode, num_iters=1)
                responses[preference] = response

        if mode == "preferences":
            responses = self.get_response(preference=None,
                                         mode="preferences",
                                         num_iters=20)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        outfile = f'{mode}_responses_{timestamp}.json'
        print(f"Saving responses to JSON file {outfile}...")

        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)

        if colab_download:
            files.download(outfile)

        return responses

    def generate_prompt_matching_class(self,
                                       preference: str,
                                       class_name: str):
        """
        I came up with the demonstrations comprising this 5-shot prompt by
        brainstorming myself.
        """
        class_name = " ".join(class_name.split("_"))

        template = dedent(f"""\
            Q: Would salad normally match a preference to eat healthy?
            A: yes

            Q: Would pork chops normally match a preference to keep kosher?
            A: no

            Q: Would bok choy normally match a preference to eat something green?
            A: yes

            Q: Would tacos normally match a preference to have something with corn?
            A: yes

            Q: Would penne alla vodka normally match a preference to be non-dairy?
            A: no

            Q: Would {class_name} normally match a preference to {preference}?
            A: """
        )

        return template


    def get_matching_classes_response(self, preference):
        """
        For each of the 101 food classes, ask the LLM whether it would normally
        satisfy a given preference, recording and returning the positive
        responses.
        """
        matching_classes = []
        for k in range(len(id_to_class_map)):
            class_name = id_to_class_map[k]
            text = self.generate_prompt_matching_class(preference, class_name)

            encodeds = self.tokenizer([text], return_tensors="pt")
            model_inputs = encodeds.to(self.device)

            generated_ids = self.model.generate(**model_inputs, max_new_tokens=2, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)

            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # separate the response from the prompt
            response_only = decoded[ len(text): ].lower().lstrip()

            if len(response_only) >= 3 and response_only[:3] == "yes":
                matching_classes.append(class_name)

        return matching_classes

    def get_matching_classes(self, preferences: list[str], colab_download=False):
        responses = {}

        for preference in tqdm(preferences):
            response = self.get_matching_classes_response(preference)
            responses[preference] = response

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        outfile = f'matching_classes_responses_{timestamp}.json'
        print(f"Saving responses to JSON file {outfile}...")

        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)

        if colab_download:
            files.download(outfile)

        return responses

def load_preferences(preferences_file):
    """
    Returns a list of preferences as strings.
    """
    with open(preferences_file) as f:
        preferences = json.load(f)

    return preferences

def load_responses(responses_file):
    """
    Returns a dictionary of lists keyed by preference strings.
    """
    with open(responses_file) as f:
        responses = json.load(f)

    return responses

def show_responses(response_dict):
    """
    Display to stdout the items in the response. Intended for viewing
        - matching classes (a.k.a. relevant classes)
        - descriptors
    """
    for preference, item_list in response_dict.items():
        print("\n\n")
        print(preference)
        print("==================")
        for item in item_list:
            print(item)


def filter_preferences(preferences: list[str], matching_classes: dict[str,list[str]], colab_download=False):
    """
    We want to filter out preferences for which at least one of the following
    is true:
        - the LLM returned an empty list of relevant foods (i.e., no matching classes)
        - the LLM returned the full set of foods as matching classes
    """
    # 1. Keep only non-empty preferences
    preferences = [preference for preference, matching_classes in matching_classes.items() if matching_classes and len(matching_classes) < len(classes)]
    print(f"{len(preferences)=}")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    outfile = f'preferences_filtered_{timestamp}.json'
    print(f"Saving responses to JSON file {outfile}...")
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(preferences, f, ensure_ascii=False, indent=4)

    if colab_download:
        files.download(outfile)

    return preferences