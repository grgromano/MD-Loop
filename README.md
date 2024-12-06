# MD-Loop
Code for [Model Discovery integrating Symbolic Regression into Sparse Identification of Nonlinear Dynamics](https://www.politesi.polimi.it/handle/10589/227268).

This repository contains the implementation of **Symbolic-SINDy**, a hybrid Model Discovery (MD) approach combining **Symbolic Regression** (SR) and **Sparse Identification of Nonlinear Dynamics** (SINDy). Developed as part of the thesis *Model Discovery integrating Symbolic Regression into Sparse Identification of Nonlinear Dynamics*, this work aims to address key limitations in existing MD methodologies, enhancing flexibility, automation, and computational efficiency.


## Abstract
Mathematical models, while essential in many diverse areas of science and engineering, are often highly complex to achieve with traditional techniques. This has motivated the search for a new, data-driven paradigm of mathematical modeling. Model Discovery (MD) is a branch of Data Science focused on the process of automatic identification of mathematical models from observed data. In other words, following a common trend nowadays, MD exploits the vast computational power and data availability to support the process of mathematical modeling. Despite MD significance, the existing techniques in this field are yet not sufficient to meet all the needs of the scientific community, often being impractical, requiring several assumptions to work, or relying on excessively user-dependent procedures. In this work, we introduce Symbolic-SINDy, a novel method that combines Symbolic Regression (SR) and Sparse Identification of Nonlinear Dynamics (SINDy) — two of the main MD approaches. Balancing strengths and limitations of SR and SINDy, Symbolic-SINDy emerges as the right compromise among the existing techniques, featuring a potentially unrestricted range of application, a practical, fast procedure and a substantial independence from human interaction. The effectiveness of the proposed method is first demonstrated on a set of classic benchmark examples in the MD field. Then, Symbolic-SINDy is used to extend MD scope in two challenging contexts: Online Model Discovery and Reduced Order Modeling.

## Installation

Clone this repository and all submodules (e.g. using `git clone --recursive`).
Python 3.6+ is recommended. Install dependencies as per [`requirements.txt`](./requirements.txt).
