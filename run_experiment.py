"""
I created a library of functions to facilitate experimentation. The main function is experiment_driver(). Eventually, I plan to
- clean this code, and remove certain inefficiencies. For example, as it stands, CLIP is asked to perform duplicate work, encoding each sample image into latent space embedding $m$ times (that is, the size of the preference pool), when these embeddings could be saved in memory after the first call.
- create a module for this library, moving it all to a seperate .py file so it can be imported instead of cluttering up a project
"""
import numpy as np
import torch
from scipy import stats
import clip
import time
import json
from google.colab import files
from tqdm import tqdm
from collections import namedtuple

def classify(im, labels, model, device):
    # image = preprocess(im).unsqueeze(0).to(device)
    image = im.unsqueeze(0).to(device) # don't need to preprocess -- dataset was
                                       # underwent preprocessing transofrmation
                                       # at load-time
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    for i, label in enumerate(labels):
        print(f"{labels[i]} = {probs[0,i]*100}%")

def score_attributes(im, attributes: list[str], model, device):
    # image = preprocess(im).unsqueeze(0).to(device)
    image = im.unsqueeze(0).to(device) # don't need to preprocess -- dataset was
                                    # underwent preprocessing transofrmation
                                    # at load-time
    text = clip.tokenize(attributes).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        out_scores = logits_per_image.cpu().numpy()

    # for i, attr in enumerate(attributes):
    #     print(f"Score \'{attributes[i]}\' = {out_scores[0, i]}")

    return out_scores

def evaluate(im, positive_attributes, model, device):
    scores = score_attributes(im, positive_attributes, model, device)
    avg_score_pos = np.average(scores)
    print(avg_score_pos)

def score_images_on_categories(imgs, categories: list[str], model, device):
    # image = preprocess(im).unsqueeze(0).to(device)
    images = imgs.to(device) # don't need to preprocess -- dataset was
                                    # underwent preprocessing transofrmation
                                    # at load-time
    categories = clip.tokenize(categories).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(categories)

        logits_per_image, logits_per_text = model(images, categories)
        out_scores = torch.flatten(logits_per_image).cpu().numpy()

    return out_scores

def score_images_on_attributes(images, attributes: list[str], model, device):
    # image = preprocess(im).unsqueeze(0).to(device)
    images = images.to(device)  # don't need to preprocess -- dataset was
                                # underwent preprocessing transofrmation
                                # at load-time

    text = clip.tokenize(attributes).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(images, text)
        out_scores = logits_per_image.cpu().numpy()

    # out_scores.shape = (100, 20)
    #                    (batch_size, num_attributes)

    # # For debugging
    # for i, attr in enumerate(attributes):
    #     print(f"Score \'{attributes[i]}\' = {out_scores[0, i]}")

    # print(out_scores.shape)

    return out_scores

def compute_avg_similarity(image_embeddings, text_inputs, model, device):
    """
    """
    tokenized_inputs = clip.tokenize(text_inputs).to(device)
    with torch.no_grad():
        # a matrix of shape t x d, containing t descriptor embeddings
        text_embeddings = model.encode_text(tokenized_inputs).to(torch.float)


    # an m x d matrix assigning each image a score per descriptor
    similarity = torch.matmul(image_embeddings, text_embeddings.T).cpu()
    
    avg_scores = torch.mean(similarity, 1)
    # print(f"{avg_scores.shape=}") # avg_scores.shape = (100,)
    
    return avg_scores, similarity


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
def get_whitelist(matching_classes):
    """
    Functionally, "handles" duplicates
    """
    K = len(classes)
    whitelist = np.zeros(K)

    indices = [class_to_id_map[food] for food in matching_classes]

    whitelist[indices] = 1

    return whitelist
#===============================================================================


################################################################################
############ Functions for Scoring Baseline Performance vs. Ours ###############
################################################################################
def score_baseline_and_ours(image_embeddings, labels_all, descriptors,
                            whitelist, preference_concept, model, device):
    """Simultaneously score baseline and our approach"""
    baseline_scores_all = []

    # 1. Score our method
    avg_scores, similarity_scores = compute_avg_similarity(image_embeddings, descriptors, model, device)

    relevance_bools_all = whitelist[labels_all].tolist()


    # 2 Score baseline
    cats = [preference_concept] # preference concept categories/classes
    baseline_scores, _baseline_similarity_scores = compute_avg_similarity(image_embeddings, cats, model, device)

    return (
        relevance_bools_all,    # new relevance labels we created
        avg_scores,             # our scores
        baseline_scores,        # baseline CLIP
        similarity_scores       # similarity score matrix, containing individual 
                                # score breakdown by descriptor for each image
        )

def compute_pearson_corr_coeff(x, y):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html
    pearson_corr = stats.pearsonr(x, y)
    stat = pearson_corr.statistic
    pvalue = pearson_corr.pvalue
    return {
        "PearsonRResult.statistic": stat,
        "PearsonRResult.pvalue": pvalue
    }

def save_results(idx, results, our_corrs, baseline_corrs,
                 intermediate=True, colab_download=False):
    """
    """

    try:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if intermediate:
            outfile = f'results_up_to_{idx}_iters_{timestamp}.json'
        else:
            # Final results
            outfile = f'ALL_{idx}_results_{timestamp}.json'

        print(f"\nSaving results to JSON file {outfile}...")

        our_corrs_summary = stats.describe(our_corrs)
        baseline_corrs_summary = stats.describe(baseline_corrs)

        try:
            corr_of_corrs_pearson = stats.pearsonr(baseline_corrs, our_corrs).statistic
        except ValueError as ve:
            # Address problem where there's a NaN value in one or both of the corrs arrays
            print(ve)
            x = np.array(baseline_corrs)
            baseline_corrs_abridged = x[np.logical_not(np.isnan(x))]
            y = np.array(our_corrs)
            our_corrs_abridged = y[np.logical_not(np.isnan(y))]
            corr_of_corrs_pearson = stats.pearsonr(baseline_corrs_abridged, our_corrs_abridged).statistic
        except Exception as e:
            print(e)
            corr_of_corrs_pearson = "Undefined"

        try:
            corr_of_corrs_spearman = stats.spearmanr(baseline_corrs, our_corrs).statistic
        except ValueError as ve:
            # Address problem where there's a NaN value in one or both of the corrs arrays
            print(ve)
            x = np.array(baseline_corrs)
            baseline_corrs_abridged = x[np.logical_not(np.isnan(x))]
            y = np.array(our_corrs)
            our_corrs_abridged = y[np.logical_not(np.isnan(y))]
            corr_of_corrs_spearman = stats.spearmanr(baseline_corrs_abridged, our_corrs_abridged).statistic
        except Exception as e:
            print(e)
            corr_of_corrs_spearman = "Undefined"

        experiment_output = {
            "Number of Tests": our_corrs_summary.nobs,
            "Our Correlations": {
                "Average": our_corrs_summary.mean,
                "Variance": our_corrs_summary.variance,
                "Min": our_corrs_summary.minmax[0],
                "Max": our_corrs_summary.minmax[1],
                "Values": our_corrs
            },
            "Baseline Correlations": {
                "Average": baseline_corrs_summary.mean,
                "Variance": baseline_corrs_summary.variance,
                "Min": baseline_corrs_summary.minmax[0],
                "Max": baseline_corrs_summary.minmax[1],
                "Values": baseline_corrs
            },
            "Correlation Between Baseline and Ours": {
                "Pearson Correlation": corr_of_corrs_pearson,
                "Spearman Correlation": corr_of_corrs_spearman
            },
            "Details by Test": results
        }
        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(experiment_output, f, ensure_ascii=False, indent=4)

        if colab_download:
            # Download from Colab to Local machine
            files.download(outfile)

        print(f"Successfully saved results to JSON file '{outfile}'.")
        return experiment_output

    except Exception as err:
        print(err)
        print("Failed to save results to JSON file.")
        return None

def get_image_embeddings_and_metadata(dataloader, test_size, model, device):
    """
    Following the docs at https://github.com/openai/CLIP explaining the API :
    > model.encode_image(image: Tensor)
    > Given a batch of images, returns the image features encoded by the vision 
    > portion of the CLIP model.
    """
    labels_all = []
    indices_all = []

    n = test_size # number of image samples in the dataset
    d = 512 # length of the image embedding vector output by CLIP
    X = torch.zeros(size=(n, d), dtype=torch.float).to(device) # placeholder matrix to store all embeddings

    print("Getting image embeddings for all samples in testset:")
    
    i = 0
    for data in tqdm(dataloader):
        images, labels, indices = data
        num_images = images.shape[0]
        index = torch.arange(i, i+num_images).to(device)

        with torch.no_grad():
            encoded_images = model.encode_image(images.to(device=device)).to(dtype=torch.float)

        X.index_copy_(0, index, encoded_images)

        i += num_images

        labels_all += labels.cpu().numpy().tolist()
        indices_all += indices.cpu().numpy().tolist()

    return (X, # a tensor of shape n x d
           labels_all,
           indices_all)


def experiment_driver(testloader,
                      test_size, 
                      preferences: list[str],
                      descriptors_list: list[list[str]],
                      relevant_classes_list: list[list[str]],
                      model, device, k_val=4):
    """
    """
    RetrievalInfo = namedtuple('RetrievalInfo', ['img_idx', 'avg_score', 'scores_by_descriptor'], defaults=(None, None, None))


    X, labels_all, indices_all = get_image_embeddings_and_metadata(testloader, test_size, model, device) # a tensor of shape n x d

    indices = torch.tensor(indices_all)

    results = []
    our_corrs = []
    baseline_corrs = []
    N = len(preferences)

    print("\nComputing similarity scores between image embeddings and descriptor embeddings for every test:")
    
    for i in tqdm(range(N)):

        preference = preferences[i]
        descriptors = descriptors_list[i]
        relevant_classes = relevant_classes_list[i]

        whitelist = get_whitelist(relevant_classes)

        score_output = score_baseline_and_ours(X, labels_all, descriptors, whitelist, preference, model, device)
        relevance_bools_all, avg_scores_all, baseline_scores_all, similarity_scores = score_output
        
        our_corr = compute_pearson_corr_coeff(relevance_bools_all, 
                                              avg_scores_all.tolist())
        baseline_corr = compute_pearson_corr_coeff(relevance_bools_all,
                                                   baseline_scores_all.tolist())
        
        top_values, top_indices = torch.topk(avg_scores_all, k=k_val, sorted=True)
        
        top_k_dataset_img_idx = indices[top_indices].tolist() # list of k indices
        top_k_scores_by_descriptor = torch.index_select(similarity_scores, 0, top_indices).tolist() # list of k lists of scores

        top_values_baseline, top_indices_baseline = torch.topk(baseline_scores_all, k=k_val, sorted=True)

        top_k_dataset_img_idx_baseline = indices[top_indices_baseline].tolist() # list of k indices

        top_k_ours = [ RetrievalInfo(img_idx, avg_score, scores)._asdict() for img_idx, avg_score, scores in zip(top_k_dataset_img_idx, top_values.tolist(), top_k_scores_by_descriptor)]
        top_k_baseline = [ RetrievalInfo(img_idx, score)._asdict() for img_idx, score in zip(top_k_dataset_img_idx_baseline, top_values_baseline.tolist())]


        res = {
            "test_id": i+1,
            "preference": preference,
            "descriptors": descriptors,
            "relevant_classes": relevant_classes,
            "baseline correlation": baseline_corr,
            "our correlation": our_corr,
            f"top_{k_val}_ours": top_k_ours,
            f"top_{k_val}_baseline": top_k_baseline
        }

        results.append(res)
        our_corrs.append(our_corr["PearsonRResult.statistic"]) # only track statistic, not pval
        baseline_corrs.append(baseline_corr["PearsonRResult.statistic"])

        # print(f"\n{i+1}. {res=}\n")

        # # Every 30 iterations, save results to a json file
        # if i % 30 == 0 and i > 0:
        #     save_results(i, results, our_corrs, baseline_corrs,
        #          intermediate=True, colab_download=True)

    # Save final results
    experiment_output = save_results(N, results, our_corrs, baseline_corrs,
                        intermediate=False, colab_download=True)

    return experiment_output

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