# ECE 285 Assignment 1

The assignment is due on April 24 2024 at 11:59pm PST.

## Overview

Assignment 1 includes 2 parts (each part is worth 50%):

1. Neural Network in NumPy
2. Classification using Neural Network

File structure:

```
assignment1
├── README.md  (this file)
├── ece285  (source code folder)
├── *.ipynb  (notebooks)
├── get_datasets.py  (download script)
└── datasets  (datasets folder)
```

## Prepare Datasets

Before you start, you need to run the following command (in terminal or in notebook beginning with `!` ) to download the datasets:

```sh
# This command will download required datasets and put it in "./datasets".
python get_datasets.py
```

## Implementation

You should run all code blocks in the following jupyter notebooks and write your answers to all inline questions included in the notebooks:

1. `neural_network.ipynb`
2. `classification_nn.ipynb`

Go through the notebooks to understand the structure. These notebooks will require you to complete the following implementations:

1. `ece285/layers/linear.py`: Implement linear layers with arbitrary input and output dimensions
2. `ece285/layers/relu.py`: Implement ReLU activation function, forward and backward pass
3. `ece285/layers/softmax.py`: Implement softmax function to calculate class probabilities
4. `ece285/layers/loss_func.py`: Implement CrossEntropy loss, forward and backward pass

You are required to go through all the modules (the ones that are already implemented for you as well) one by one to to understand the structure. This will help you when transitioning to deep learning libraries such as PyTorch. Here are some files that you should go through that are already implemented for you:

1. `ece285/layers/sequential.py`
2. `ece285/utils/trainer.py`
3. `ece285/utils/optimizer.py`

## Submission

### Things to keep in mind:

1. Edit **only** the parts that are asked of you. Do **not** change the random seed where it is set. This might not make it possible to get similar results.
2. Try to avoid `for` loops in all the implementations, the assignment can be solved without any for loops (vectorized implementation).
3. **Only answer questions and plot figures at givens cells**. TAs may miss your answer if you put it at the wrong cell.

After you complete all the functions and questions, you should upload the following files to Gradescope:

1. 4 `*.py` files that you have to implement (linear, relu, softmax, loss_func).
2. 2 notebook source `*.ipynb` files.
3. Exported PDF of notebooks (You could export jupyter notebook as PDF in web portal, do not need to install LaTeX packages) and merge them.

You should organize files like this:

1. Put all source files (7 python files and 5 notebooks) into a single `.zip` file then upload it on Gradescope.
2. Merge exported PDF of five notebooks into a single PDF file and upload it on Gradescope. Make sure to select correct page numbers for each question in Gradescope, otherwise TAs might miss the answers.

## Appendix

### How to Use Jupyter Notebook

Part of our skeleton code is provided using Jupyter notebook(`*.ipynb`). So there we provide a short instruction about how to use Jupyter Notebook.

Jupyter notebook is a powerfull interactive developement tool. Here we could Jupyter notebook file as notebook for short.

Jupyter notebook is pre-installed on Datahub, you can just use it via web-portal. You could also install it on your local machine. Here is the official install instruction([https://jupyter.org/install](https://jupyter.org/install)), which also included how to run it via terminal.

You may find on Jupyter website, they provide Jupyter Lab. Jupyter Lab is basically a new version of Jupyter notebook with more features and different interface. The basic usage is almost the same.

All notebooks are made up with multiple blocks. There are different kinds of blocks, the most common blocks are:

- Code block
- Markdown block

#### Code

For code block, you can write python code in code block, after finishing your code you could press run bottom on the jupyter note interface(normally on the top of the web interface). You can also use `Ctrl + Enter` or `Shift + Enter` to execute the block.

After you execute a block, Jupyter Notebook will execute your code with python and store all the function and variable you defined in memory. So you could still use those variables and function is other blocks.

For code blocks, you can think of jupyter notebook as a python console with an interface.

#### Markdown

Markdown block is where you can write some text.

When you execute Markdown block(the same method as Code block), your text will be compiled using Markdown grammar, instead of executing it with python.

In our assignment, we put some inline questions in notebooks, you are supposed to answer them with text.

#### Conclusion

In all, notebooks are basically some combination of code and text.

### How to Set Up Python Environment (Optional)

If you run the assignment on Datahub, you do **not** need to set up python environment. Our code is tested on Datahub.

If you run the assignment on your local machine or other environment and meet some problems, you **may** need to set up python environment.

We prepare a `requirements.txt` file (same versions as the datahub packages) for you to install python packages. You can install the packages by running the following command in terminal:

```sh
pip install -r requirements.txt
```

This should solve most package issues. But if you still have some problems, we recommend you to use conda environment. You can install anaconda or miniconda by following the instruction on [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html). After you install it, you can run the following command to set up python environment:

```sh
conda create -n ece285 python=3.9.5  # same python version as the datahub
conda activate ece285
pip install -r requirements.txt
```

If you have any questions, feel free to contact TAs.

### Other Tips

How to unzip a zip file:

```sh
unzip XX.zip  # in terminal or in notebook beginning with `!`
```
