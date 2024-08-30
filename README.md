# Exploring Semantic Coexistentence in Open-Vocabulary Segmentation

Semantic segmentation is essential in real-world appli-
cations such as autonomous driving. Open-vocabulary se-
mantic Segmentation models advance conventional methods
by extending pixel label classes to any arbitrary sets of text
labels. One recent two-stage approach relies on a class-
agnostic mask model and a pretrained vision-language
classifier to assign a text label to each mask proposal flex-
ibly. However, this pipeline fails to consider the implicit
co-existing relationship of the segmentation targets for more
accurate segmentation. This paper proposes CoSeg, a novel
open-vocabulary segmentation modeling framework with a
pretrained vision-language model and referral segmenta-
tion architecture that exploits semantic coexistence in the
joint visual-linguistic space. Despite a simple architecture,
our no-training and fully-supervised models achieve com-
petitive performance in a cross-dataset evaluation, espe-
cially in a contextually rich environment. We believe our
method establishes a foundation for future exploration of
semantic modeling in images.

## Set up Environment

Docker Image @ docker://tovitu/coseg:latest
