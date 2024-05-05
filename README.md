# TasteRank
Read the (unpublished) research [paper](TasteRank-paper.pdf).

__Abstract__
How can we find relevant images in a database when queries are arbitrary user preferences such as “get protein from plants”? To compute relevance scores for each image, I investigate following a two-step framework: (1) first break down the target preference into descriptive features, then (2) perform recognition on the images over these features. The first step produces a list of descriptors for a given preference by calling on a language model, outputting, for example, [Legumes, Nuts & seeds, Leaved greens, Tofu & tempeh]. Step two leverages CLIP to test for the presence of the descriptors in zero-shot fashion. Then, a final score is computed for each image capturing the extent to which
the descriptors were detected, and the score is treated as an approximation for relevance. For relevant-image identification, the proposed approach outperforms the baseline and
comes built-in with a mechanism to explain its decisions. As an alternative to traditional tag-based retrieval schemes, the method has the potential to provide more sophisticated
personalization.

The code to replicate experiments is in the Jupyter Notebook [TasteRank.ipynb](TasteRank.ipynb). It is highly recommended to run the notebook to Google Colab, noting well the runtime requirements for each section of the notebook.