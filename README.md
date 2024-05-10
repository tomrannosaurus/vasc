What is Autoencoder?
In class, we discussed the feedforward neural network, and autoencoders are a special type of feedforward neural network. The autoencoder algorithm comprises two main elements: the encoder and the decoder. 

[insert graph]

The encoder transforms the input data into a reduced dimensional representation, which is referred to as “latent space”; from the latent space, the decoder then reconstructs the output, which ideally should be as close as the input data. 

Broadly, the loss function here is usually the reconstruction loss, which is the difference between the original input and the reconstructed output {/}. 

What is variational encoder
Before we jump into the neural network in our paper, we first need to talk about the foundation of that algorithm, which is a specific type of autoencoder, the variational encoder (VAE).

Variational autoencoder was proposed in 2013 by Diederik P. Kingma and Max Welling at Google and Qualcomm {cite}. The variational encoder is a special type of auto encoder that provides a probabilistic manner for describing an observation in latent space. Rather than a single encoding vector for latent space, VAEs model two different vectors: a vector of means, “$$/mu$$,” and a vector of standard deviations, “$$\sigma$$”. Thus, rather than building an encoder that outputs a single value to describe each latent state, we formulate our encoder to describe a probability distribution for each latent attribute. Therefore, VAEs allow for interpolation and random sampling, which greatly expand their capabilities.

[insert graph]

Math behind VAE
Now we understand the basic structure of VAE, lets take deeper dive into the math.
We want to generate a distribution of $$z$$ from the observational data $$x$$, intuitively, we would do this:
$$P(z|X) =\frac{P(X|z)P(z)}{P(X)}$$
However, the calculation of $$P(z|X)$$ is intractable.
Therefore, we want to approximate $$P(x)$$ using the variational probability $$Q(z|X)$$ by minimizing the KL divergence:

$$D[Q(z \mid X) \| P(z \mid X)] = E_{z \sim Q} [\log Q(z \mid X) - \log P(z \mid X)]$$

By applying Bayes’ rule:
$$\log P(X) - D[Q(z \mid X) \| P(z \mid X)] = E_{z \sim Q} [\log P(X \mid z) - D[Q(z) \| P(z)]]$$

where
$$P(X)$$ is constant
$$E_{z \sim Q}$$ represents the expectation over z that is sampled from $$Q$$.
minimizing the KL divergence is equivalent to maximizing the right-hand part of Equation
The right-hand part has a natural autoencoder structure, with the encoder $Q(z|X)$ from $X$ to $z$ and the decoder $P(X|z)$ from $z$ to $X$.

Intuition of the paper
To get to the meat of the paper, lets switch gears a little bit. Let’s talk a bit about biology and how this algorithm was motivated by RNA sequencing.
RNAseq is a powerful tool for understanding the molecular mechanisms of cancer development and developing novel strategies for cancer prevention and treatment. There’s generally two types of RNA sequencing techniques, the bulk RNA sequencing and single cell RNA sequencing.
Bulk RNA sequencing measures the average gene expression across a population of heterogenous cells. Before single-cell sequencing arrived, bulk sequencing was the preferred method. It enabled the study of the genome (DNA) and transcriptome (RNA), among other omics. With bulk RNA sequencing, you can compare the results of the patients with lung cancer with those that are healthy. However, the answers to that question may lie behind certain cell types. This requires looking at the expression of genes in individual cells instead of an average representation. In that case, single-cell sequencing provides the potential to find molecular differences which are only linked to specific cell types.

[insert graph]
Motivation of the paper 
A comprehensive characterization of the transcriptional status of individual cells enables us to gain full insight into the interplay of transcripts within single cells. However, scRNA-seq measurements typically suffer from large fractions of observed zeros, where a given gene in a given cell has no unique molecular identifiers or reads mapping to it. Therefore, a new method other than the current methods, which have limitation in its performance, is needed. Therefore, this paper propose a new method: deep variational autoencoder for scRNA-seq data (VASC).

Math behind VASC

Unlike VAE, which generally only has three steps, VASC can be broken down into 6 steps: 

[replace figure with horizontal drawing]



########## TOM WORKING ON THIS SECTION ##########

Input layer: the input layer uses the expression matrix from scRNA-seq data. The data were log-transformed and re-scaled to make the results more robust. 

Relevant code:
```python
expr[expr<0] = 0.0
if log:
    expr = np.log2( expr + 1 )
if scale:
    for i in range(expr.shape[0]):
        expr[i,:] = expr[i,:] / np.max(expr[i,:])
```


Dropout layer: A dropout layer was added immediately after the input layer, with the dropout rate set as 0.5. This layer sets some features to zeros during the encoding phase, to increase the performance in model learning. This layer forces subsequent layers to learn to avoid dropout noises.

Relevant code:
```python
h0 = Dropout(0.5)(expr_in)
```


Encoder network: The encoder network was designed as a three-layer fully-connected neural network with decreasing dimensions 512, 128, and 32. L1-norm regularization was added for the weights in this layer, which penalized the sparsity of the model. The next two layers were accompanied by ReLU activation, which made the output sparse and stable for deep models. 

Latent Sampling: Latent variables $z$ were modeled by a Gaussian distribution, with the standard normal prior $N(0,I)$. Usually, both the parameters $l$ and $/Sigma$ needed to be estimated, with a linear activation used to estimate $/mu$. The authors used a ‘softplus’ activation was used for the estimation of $/Sigma$.

Decoder network: The decoder network used the generated z to recover the original expression matrix, which was designed as a three-layer fully- connected neural network with dimensions of hidden units 32, 128, and 512, respectively, and an output layer. The first three layers used ‘ReLU’ activations and the final layer with sigmoid to make the output within [0,1].

ZI layer: An additional ZI layer was added after the decoder network. The authors modeled the dropout events by the probability $e^{-\tilde{y}^2}}$. Since back-propagation cannot deal with stochastic units, new approach was proposed. A Gumbel- softmax distribution, 
$$s = \frac{\exp\left(\frac{\log p + g_0}{\tau}\right)}{\exp\left(\frac{\log p + g_0}{\tau}\right) + \exp\left(\frac{\log q + g_1}{\tau}\right)}$$ was introduced to overcome this issue, where $p $is the probability for dropout and $q=1-p$,$g0$,$g1$ were sampled from a Gumbel (0,1) distribution. 

The samples could then be obtained by first drawing an auxiliary sample $u \sim Uniform (0,1)$ and then computing $g=-log(-log u)$

Loss function
There’s two parts of the loss function. The first part was computed by binary cross-entropy loss function, since the scale of the data is [0,1]. The second part is the same as minimizing the Kullback–Leibler (KL) divergence in VAE.

Optimization
The whole structure could be optimized end-to-end using the stochastic gradient descent-based optimization algorithm that we learned in class.

########## END TOM WORKING ON THIS SECTION ##########

Experiments from Paper





Conclusions 




References

