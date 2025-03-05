# Yuhang-Zhang
Please download the image dataset from the official website, divide it into training, validation, and test sets, and then run this project.
Initially, only the classifiers are trained using logistic regression, with a batch size of 16, a weight decay of 1×10-8, and a learning rate of 1.
Then, the entire network is fine-tuned using stochastic gradient descent, with a batch size of 64, a weight decay of 1×10-6, and a learning rate of 1×10-2.
