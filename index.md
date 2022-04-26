## Welcome to GitHub Pages of SparseEvoAttack

Reproduce our results: [GitHub](https://github.com/SparseEvoAttack/SparseEvoAttack.github.io) 

Check out our paper: [Query Efficient Decision Based Sparse Attacks Against Black-Box Machine Learning Models](https://arxiv.org/abs/2202.00091)

#### ABSTRACT

Despite our best efforts, deep learning models remain highly vulnerable to even tiny adversarial perturbations applied to the inputs. The ability to extract information for solely the output of a machine learning model to craft adversarial perturbations to black-box models is a practical threat against real-world systems, such as autonomous cars or machine learning models exposed as a service (MLaaS). Of particular interest are sparse attacks. The realization of sparse attacks in black-box models demonstrates that machine learning models are more vulnerable than we believe.  Because, these attacks aim to minimize the number of perturbed pixels—measured byl0norm—required to mislead a model by solely observing the decision (the predicted label) returned to a model query; the so-called decision-based attack setting.  But, such an attack leads to an NP-hard optimization problem. We develop an evolution-based algorithm—SparseEvo—for the problem and evaluate against both convolutional deep neural networks and vision transformers. Notably, vision transformers are yet to be investigated under a decision-based at-tack setting. SparseEvo requires significantly fewer model queries than the state-of-the-art sparse attack Pointwise for both untargeted and targeted attacks.  The attack algorithm, although conceptually simple, is also competitive with only a limited query budget against the state-of-the-art gradient-based Whitebox attacks in standard computer vision tasks such as ImageNet. Importantly, the query efficient SparseEvo, along with decision-based attacks, in general, raise new questions regarding the safety of deployed systems and poses new directions to study and understand the robustness of machine learning models.

#### ALGORITHM DIAGRAM
![Figure 1](image/SpaEvo Algo-diagram horizon.svg#gh-dark-mode-only)

Figure 1: An illustration of SparseEvo algorithm. Population Initialization creates the first population generation. This population is evolved over iterations through Binary Differential Recombination, Mutation, Fitness Evaluation (Adversarial Example Construction and Fitness Computation) and Selection stages. The source and starting images (used in a targeted attack) are employed to create the initial candidate solutions —binary vector representations—at Population Initialisation and to construct an adversarial example based on a candidate solution v(m) at Fitness Evaluation stage.

#### VISUALIZATION

![Figure 2](image/gh-sparse result visualization.svg#gh-dark-mode-only)

Figure  2:   Targeted  Attack. Malicious instances generated for a  sparse attack with different query budgets using our SparseEvo attack algorithm employed on black-box models built for the ImageNet task. With an extremely sparse perturbation (_78 perturbed pixels over a total of 50,176pixels_), an image with ground-truth label **traffic light** misclassified as a **street sign**.

![Figure 3](image/gh-sparse result visualization-2.svg#gh-dark-mode-only)

Figure 3: Targeted Attack. Malicious instances generated for a sparse attack with different query budgets using our SparseEvo attack algorithm employed on black-box models built for the ImageNet task.
