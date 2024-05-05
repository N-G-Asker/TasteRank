"""
Modifies Experiment Driver to enable taking a deeper look at individual experiments.

Adds functionality to return info on the top-4 results for each test.

TODO: change this to have more flexible top-k reporting info, rather than hardcoing 4.
"""
from run_experiment import *

def score_baseline_and_ours(dataloader, descriptors,
                            whitelist, preference_concept):
    """Simultaneously score baseline and our approach"""
    relevance_bools_all = []
    labels_all = []
    avg_scores_all = []
    baseline_scores_all = []
    indices_list = []

    for data in dataloader:
        # 1. Score our method on this batch
        images, labels, indices = data
        avg_scores = evaluate_images(images, descriptors)


        avg_scores_all += avg_scores.tolist() # convert from np arrays
        labels_all += labels.cpu().numpy().tolist()
        relevance_bools_all += whitelist[labels].tolist()


        # 2 Score baseline on this batch
        imgs = torch.squeeze(images, dim=0) # remove first dimension

        cats = [preference_concept] # preference concept categories/classes
        scores = score_images_on_categories(imgs, cats)

        baseline_scores_all += scores.tolist() # convert from np arrays

        indices_list += indices

    return (
        relevance_bools_all,    # new relevance labels we created
        avg_scores_all,         # our scores
        baseline_scores_all,     # baseline CLIP
        indices_list
        )

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

    for i, attr in enumerate(attributes):
        print(f"Score \'{attributes[i]}\' = {out_scores[0, i]}")

    return out_scores

def evaluate(im, positive_attributes, model, device):
    scores = score_attributes(im, positive_attributes, model, device)
    avg_score_pos = np.average(scores)
    print("Average Score", avg_score_pos)
    # return scores

def experiment_driver(testloader_with_idxs,
                      preferences: list[str],
                      descriptors_list: list[list[str]],
                      relevant_classes_list: list[list[str]]):
    """
    Altered to return top 4 highest-scoring images, together with their scores
    """
    results = []
    our_corrs = []
    baseline_corrs = []
    top4 = []
    N = len(preferences)
    for i in tqdm(range(N)):

        preference = preferences[i]
        descriptors = descriptors_list[i]
        relevant_classes = relevant_classes_list[i]

        whitelist = get_whitelist(relevant_classes)

        score_output = score_baseline_and_ours(testloader_with_idxs, descriptors, whitelist, preference)
        relevance_bools_all, avg_scores_all, baseline_scores_all, indices = score_output

        our_corr = compute_pearson_corr_coeff(relevance_bools_all, avg_scores_all)
        baseline_corr = compute_pearson_corr_coeff(relevance_bools_all, baseline_scores_all)

        res = {
            "test_id": i+1,
            "preference": preference,
            "descriptors": descriptors,
            "relevant_classes": relevant_classes,
            "baseline correlation": baseline_corr,
            "our correlation": our_corr
        }

        avg_scores_all = np.array(avg_scores_all)
        baseline_scores_all = np.array(baseline_scores_all)
        indices = np.array(indices)

        top_4_score_idx = np.argpartition(avg_scores_all, -4)[-4:]
        top_4_dataset_idx = indices[top_4_score_idx]
        top_4_scores = avg_scores_all[top_4_score_idx]

        top_4_b_score_idx = np.argpartition(baseline_scores_all, -4)[-4:]
        top_4_b_dataset_idx = indices[top_4_b_score_idx]
        top_4_b_scores = baseline_scores_all[top_4_b_score_idx]

        top_4 = (
            tuple(zip(top_4_dataset_idx, top_4_scores)),
            tuple(zip(top_4_b_dataset_idx, top_4_b_scores))
            )

        top4.append(top_4)

        results.append(res)
        our_corrs.append(our_corr["PearsonRResult.statistic"]) # only track statistic, not pval
        baseline_corrs.append(baseline_corr["PearsonRResult.statistic"])

        print(f"\n{i+1}. {res=}\n")

        # Every 30 iterations, save results to a json file
        if i % 30 == 0 and i > 0:
            save_results(i, results, our_corrs, baseline_corrs,
                 intermediate=True, colab_download=True)

    # Save final results
    experiment_output = save_results(N, results, our_corrs, baseline_corrs,
                        intermediate=False, colab_download=True)

    return experiment_output, top4