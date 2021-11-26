## Welcome to GitHub Pages of SparseEvoAttack

[View on GitHub](https://github.com/SparseEvoAttack/SparseEvoAttack.github.io/blob/main/index.md) 

#### ABSTRACT

Despite our best efforts, deep learning models remain highly vulnerable to eventiny adversarial perturbations applied to the inputs. The ability to extract informa-tion formsolelythe output of a machine learning model to craft adversarial pertur-bations to black-box models is apracticalthreat against real-world systems, suchas autonomous cars or machine learning models exposed as a service (MLaaS). Of particular interest aresparse attacks. The realisation of sparse attacks in black-box models demonstrates that machine learning models are more vulnerable thanwe believe.  Because, these attacks aim tominimize the number of perturbed pix-els—measured byl0norm—required to mislead a model bysolelyobserving thedecision (the predicted label) returned to a model query; the so-calleddecision-based attack setting.  But, such an attack leads to an NP-hard optimization prob-lem. We develop an evolution-based algorithm—SparseEvo—for the problem andevaluate against both convolutional deep neural networks andvision transformers. Notably, vision transformers are yet to be investigated under a decision-based at-tack setting. SparseEvo requires significantly fewer model queries than the state-of-the-art sparse attack Pointwise for both untargeted and targeted attacks.  The attack algorithm, although conceptually simple, is also competitive with only a limited query budget against the state-of-the-art gradient-based whitebox attacks in standard computer vision tasks such asImageNet. Importantly, the query ef-ficient SparseEvo, along with decision-based attacks, in general, raise new ques-tions regarding the safety of deployed systems and poses new directions to study and understand the robustness of machine learning models.

#### VISUALIZATION

![Figure 1](image/gh-sparse result visualization.svg#gh-dark-mode-only)

Figure  1:   Targeted  Attack. Malicious  instances  generated  for  a  sparse  attack  with  different query budgets using our SparseEvo attack algorithm employed on black-box models built for the ImageNet task. With an extremely sparse perturbation (_78 perturbed pixels over a total of 50,176pixels_), an image with ground-truth label **traffic lightis** misclassified as a **street sign**.

![Figure 2](image/gh-sparse result visualization-2.svg#gh-dark-mode-only)

Figure 2: Targeted Attack. Malicious instances generated for a sparseattack with different query budgets using our SparseEvo attack algorithm employed on black-boxmodels built for the ImageNet task.
