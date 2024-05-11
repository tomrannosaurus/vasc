<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script> 
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>




-------------------------------------------------------------


# What is Autoencoder?
In class, we discussed the feedforward neural network, and autoencoders are a special type of feedforward neural network. The autoencoder algorithm comprises two main elements: the encoder and the decoder. 



The encoder transforms the input data into a reduced dimensional representation, which is referred to as “latent space”; from the latent space, the decoder then reconstructs the output, which ideally should be as close as the input data. 

Broadly, the loss function here is usually the reconstruction loss, which is the difference between the original input and the reconstructed output (1).

What is the variational encoder?
Before we jump into the topic of our paper, we first need to talk about the foundation of that algorithm, which is a specific type of autoencoder, the variational encoder (VAE).

Variational autoencoder was proposed in 2013 by Diederik P. Kingma and Max Welling at Google and Qualcomm (2). The variational encoder is a special type of auto encoder that provides a probabilistic manner for describing an observation in latent space. Rather than a single encoding vector for the latent space, VAEs model two different vectors: a vector of means, “$$\mu$$,” and a vector of standard deviations, “$$\sigma$$”. This way, the VAEs allow us to interpolate and use random samples, which greatly expands their capabilities.

Math behind VAE
Now that we understand the basic structure of VAE, let's dive deeper into the math (3).
We want to generate a distribution of the latent space $$z$$ from the observational data $$x$$, intuitively, we would do this:
$$P(z|X) =\frac{P(X|z)P(z)}{P(X)}$$
However, the calculation of $$P(z|X)$$ is intractable.
Therefore, we want to approximate $$P(z|X)$$ using the variational probability $$Q(z|X)$$ by minimizing the Kullback Leibler (KL) Divergence:

$$D[Q(z \mid X) \| P(z \mid X)] = E_{z \sim Q} [\log Q(z \mid X) - \log P(z \mid X)]$$

By applying Bayes’ rule:
$$\log P(X) - D[Q(z \mid X) \| P(z \mid X)] = E_{z \sim Q} [\log P(X \mid z) - D[Q(z) \| P(z)]]$$

where
$$P(X)$$ is constant
$$E_{z \sim Q}$$ represents the expectation over $$z$$ that is sampled from $$Q$$.
minimizing the KL divergence is equivalent to maximizing the right-hand part of the Equation
The right-hand part has a natural autoencoder structure, with the encoder $$Q(z|X)$$ from $$X$$ to $$z$$ and the decoder $$P(X|z)$$ from $$z$$ to $$X$$.

Intuition of the paper
Before we learn the method of this paper, let's switch gears a bit. We need to know what this method is inspired by and what problem are the authors trying to solve. 
RNA sequencing
RNA sequencing is a powerful tool for understanding the molecular mechanisms of cancer development and developing novel strategies for cancer prevention and treatment. There are generally two types of RNA sequencing techniques: bulk RNA sequencing and single-cell RNA sequencing (scRNA-seq) (4). Bulk RNA sequencing measures the average gene expression across the population of various cells. With bulk RNA sequencing, you can compare the results of lung cancer patients with those of healthy ones (5). 
However, the answers to that question may lie behind certain cell types.

Here is an analogy: bulk RNA sequencing is like a glass of smoothie; it has all kinds of fruits and vegetables, same as a blood sample, which is a mixture of different kinds of cell types, e.g., B cells, T cells. If we are particularly interested in the flavor characteristics of raspberry, it is difficult to do that with a glass of smoothie mixed with bananas, oranges, and pineapples. Therefore, this requires looking at the expression of genes in individual cells instead of an average representation. In that case, single-cell sequencing provides the potential to find molecular differences that are only linked to specific cell types.
The motivation of the paper 
As we learned from the analogy earlier, scRNA-seq provided a way to comprehensively characterize individual cells' transcriptional information, enabling us to gain full insight into the interplay of the transcripts. 

However, scRNA-seq measurements typically suffer from large fractions of observed zeros, where a given gene in a given cell has no unique molecular identifiers or reads mapping to it. This is a very difficult challenge when it comes to dimension reduction. Therefore, this paper proposed a new method: deep variational autoencoder for scRNA-seq data (VASC) to tackle this issue. 

Math & Code Behind VASC

Unlike VAE, which generally only has three steps, VASC can be broken down into 6 steps: 



########## TOM WORKING ON THIS SECTION ##########

### Input layer

The input layer uses the expression matrix from scRNA-seq data. The input layer handles three major transformations: ensuring non-negativity, log-transformation, and re-scaling. These transformations are enabled by default given the algorithm's specific application for RNA-seq data for which these transformations are typically applied. The log-transformation and scaling help compress the dynamic range of the input data which can help with optimization in a non-convex space such as a neural network.

```python
expr[expr<0] = 0.0
if log:
    expr = np.log2( expr + 1 )
if scale:
    for i in range(expr.shape[0]):
        expr[i,:] = expr[i,:] / np.max(expr[i,:])
```


### Dropout layer 

A dropout layer is added immediately after the input layer, with the dropout rate set as 0.5. This layer randomly sets some features to zeros during the encoding phase, to increase the performance in model learning. This layer forces subsequent layers to learn to avoid dropout noises. Mathematically, if we denote the input to the dropout layer as $$ h_0 $$, and the output as $$ h_{\text{drop}_0} $$, then: $$ h_{\text{drop}_0}[i,j] = h_0[i,j] \times m[i,j] $$ where $$ m[i,j] \sim \text{Bernoulli}(0.5) $$ for each $$ i, j $$. In other words, each element of $$ h_0 $$ is independently set to 0 with 50% probability.(6)

```python
h0 = Dropout(0.5)(expr_in)
```


### Encoder network

The encoder network is designed as a three-layer fully-connected neural network with decreasing dimensions 512, 128, and 32. L1-norm regularization was added for the weights in this layer, which penalizes the sparsity of the model. The next two layers are accompanied by ReLU activation, which allows the model to learn complex non-linear relationships.

```python
h1 = Dense( units=512,name='encoder_1',kernel_regularizer=regularizers.l1(0.01) )(h0)
h2 = Dense( units=128,name='encoder_2' )(h1)  
h3 = Dense( units=32,name='encoder_3' )(h2_relu)
```





### Latent Sampling 
The latent sampling layer generates samples from the learned latent representation. This layer takes the output from the encoder network and uses it to parameterize a probability distribution in the latent space, from which samples are then drawn. The latent variables $$ z $$ are modeled using a multivariate Gaussian distribution. The encoder network outputs the parameters of this distribution: the mean vector $$ \mu $$ and the covariance matrix $$ \Sigma $$ or, if the `var` flag is enabled, the log-variances which are exponentiated to get the diagonal of $$ \Sigma$$ .

Mathematically, the latent distribution is $$ z \sim \mathcal{N}(\mu, \Sigma) $$ where $$ \mu = \mu(X) $$  and $$ \Sigma = \Sigma(X) $$ are functions of the input data X, learned by the encoder network. In the code, the latent sampling layer is implemented as follows.

```python
z_mean = Dense( units= self.latent ,name='z_mean' )(h3_relu)
```
The line above uses a fully-connected layer to map the output of the encoder (h3_relu) to the mean vector $$ \mu $$ of the latent Gaussian distribution. 

```python
if self.var:
    z_log_var = Dense( units=2,name='z_log_var' )(h3_relu)
    z_log_var = Activation( 'softplus' )(z_log_var)
    z = Lambda(sampling, output_shape=(self.latent,))([z_mean,z_log_var])
else:
    z = Lambda(sampling, output_shape=(self.latent,))([z_mean])
```
The code above handles the sampling process. If the `var` parameter is True, another fully-connected layer is used to output the log-variances of the Gaussian, which are then passed through a softplus activation to ensure they are positive (since variances must be non-negative). The sampling function is then called with both $$ \mu $$ and $$ \Sigma $$. (If 'var' is False, the sampling function is called with only $$ \mu $$, and a fixed unit variance is used.)

```python
def sampling(args):
    epsilon_std = 1.0

    args = tf.convert_to_tensor(args)

    if args.shape[0] == 2:
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean),
                              mean=0.,
                              stddev=epsilon_std)
    #
        return z_mean + K.exp( z_log_var / 2 ) * epsilon
    else:
        z_mean = args[0]
        epsilon = K.random_normal(shape=K.shape(z_mean),
                              mean=0.,
                              stddev=epsilon_std)
        return z_mean + K.exp( 1.0 / 2 ) * epsilon
```

The function above implements a sampling reparameterization trick, which allows for more efficient gradient computation during training. Instead of directly sampling from $$ \mathcal{N}(\mu, \Sigma) $$, it samples a standard Gaussian noise $$\varepsilon \sim \mathcal{N}(0, 1) $$ and then computes $$ z =  \mu + \Sigma^{1/2} \dot \varepsilon $$. This is equivalent to sampling from $$ \mathcal{N}(\mu, \Sigma) $$ but allows for backpropagation through the sampling step.


### Decoder network

The decoder network maps the latent representation $$z$$ back to the original gene expression space, thereby reconstructing the input data. It is a mirror image of the encoder network. It consists of a three-layer fully-connected neural network with dimensions of hidden units 32, 128, and 512, respectively, and an output layer. The decoder transforms the ​​latent representation into a higher-dimensional output that matches the dimensions of the input gene expression profile.

```python
decoder_h1 = Dense( units=32,name='decoder_1' )(z)
decoder_h2 = Dense( units=128,name='decoder_2' )(decoder_h1_relu)
decoder_h3 = Dense( units=512,name='decoder_3' )(decoder_h2_relu)
expr_x = Dense(units=self.in_dim,activation='sigmoid')(decoder_h3_relu)
```
The final layer of the decoder maps the output of decoder_h3 to the original dimension of the gene expression profile. 

Mathematically, if we call the function represented by the decoder network as $$ g(\cdot) $$, then $$ \hat{X} = g(z) $$ when $$ \hat{X} $$ is the reconstructed gene expression profile. The goal of the decoder is to make $$ \hat{X} $$ as close as possible to the original input $$ X $$. This is achieved through the optimization of the reconstruction term in the loss function (a mix of KL divergence and binary cross-entropy between $$ X $$ and $$ \hat{X} $$).


### ZI layer 

An additional ZI layer is added after the decoder network, and could be considered the VASC ‘secret sauce.’ The ZI layer models the dropout events by setting some decoded expression values to zero based on a double exponential distribution. The dropout events are modeled by the probability $$p_{ij} = e^{-\tilde{y}^2} $$. 

```python
def sampling_gumbel(shape,eps=1e-8):
    u = K.random_uniform( shape )
    return -K.log( -K.log(u+eps)+eps )

def compute_softmax(logits,temp):
    z = logits + sampling_gumbel( K.shape(logits) )
    return K.softmax( z / temp )

def gumbel_softmax(args):
    logits,temp = args
    return compute_softmax(logits,temp)
```

Since back-propagation cannot deal with stochastic units, new approach was proposed. A Gumbel- softmax distribution, $$s = \frac{\exp\left(\frac{\log p + g_0}{\tau}\right)}{\exp\left(\frac{\log p + g_0}{\tau}\right) + \exp\left(\frac{\log q + g_1}{\tau}\right)}$$ was introduced to overcome this issue, where $$p $$ is the probability for dropout and $$q=1-p$$,$$g_0$$,$$g_1$$ were sampled from a $$Gumbel \sim (0,1)$$ distribution. The samples could then be obtained by first drawing an auxiliary sample $$u \sim Uniform (0,1)$$ and then computing $$g=-log(-log u)$$

```python
expr_x_drop = Lambda(lambda x: -x ** 2)(expr_x)
expr_x_drop_p = Lambda( lambda x:K.exp(x) )(expr_x_drop)
expr_x_nondrop_p = Lambda( lambda x:1-x )( expr_x_drop_p )
logits = merge( [expr_x_drop_log,expr_x_nondrop_log],mode='concat',concat_axis=-1 )
samples = Lambda( gumbel_softmax,output_shape=(self.in_dim,2,) )( [logits,temp_] )   
```

In the code above we see `expr_x` is the output of the decoder network (without the sigmoid activation). `expr_x_drop` computes $$-y_{ij}^2$$, and `expr_x_drop_p` computes the dropout probability $$p_{ij}$$. `expr_x_nondrop_p` computes the probability of not being a dropout $$(1 - p_{ij})$$. The code then computes the log-probabilities and concatenates them. These log-probabilities are used to compute the final output of the ZI layer using the Gumbel-Softmax trick. Said another way, the Gumbel-Softmax trick is used as a differentiable approximation to sampling from a discrete distribution, enabling gradient backpropagation through the ZI layer.

In sum, the ZI layer models the dropout events in the input RNA-seq data by estimating a dropout probability for each gene based on the latent representation $$z$$. It then uses the Gumbel-Softmax trick to sample binary “dropout masks” that are multiplied element-wise with the decoder output.

### Loss function

There are two parts of the loss function. The first part was computed by binary cross-entropy loss function, since the scale of the data is [0,1], which can be seen to be the ‘reconstruction loss.’ Binary cross-entropy can be interpreted as the probability of each gene being expressed.

The second part is the same as minimizing the Kullback–Leibler (KL) divergence typically seen in VAE. The reconstruction loss ensures that the autoencoder is able to effectively reconstruct the input data from the latent representation, while the KL divergence loss acts as a regularizer, encouraging the learned latent distribution to be close to a prior distribution (assumed to be a standard Gaussian).

```python
def vae_loss(self, x, x_decoded_mean):
    xent_loss = in_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    if var_:
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    else:
        kl_loss = - 0.5 * K.sum(1 + 1 - K.square(z_mean) - K.exp(1.0), axis=-1)
    return K.mean(xent_loss + kl_loss)
```

Mathematically, the total loss for a single sample can be written as:

$$
L = - E_{q(z|x)}[\log p(x|z)] + \text{KL}(q(z|x) \parallel p(z))
$$

where $$ q(z|x) $$ is the learned latent distribution (the encoder output), $$ p(x|z) $$ is the decoder output distribution, and $$ p(z) $$ is the prior distribution on the latent space. The first term is the negative expected log-likelihood of the data under the decoder output distribution, which corresponds to the reconstruction loss. The second term is the KL divergence between the learned latent distribution and the prior, which acts as a regularizer.


Optimization
The entire structure as delineated above is optimized by a variant of the stochastic gradient descent optimization algorithm that we learned in class known as ​​RMSprop. RMSprop has an adaptive learning rate (analogous to momentum) for each parameter similar to Adam, but without the frictional component. It should also be noted that the algorithm uses batch processing to avoid overfitting and promote faster learning.

```python
opt = RMSprop( lr=0.0001 )
vae.compile( optimizer=opt,loss=None )
```
This process is repeated for a fixed number of iterations, or until a stopping criterion is met. The training stops if there is no obvious (greater than 0.1) decrease in the loss function within a given number of epochs. In terms of code, the training loop is set to run for a maximum of 'epoch' iterations, but will stop early if the loss does not improve significantly for 'patience' epochs in a row.

```python
        if e % patience == 1:
            print("Epoch %d/%d" % (e, epochs))
            print("Loss:" + str(train_loss))
            print("current loss: ", cur_loss, "previous loss:", prev_loss)
            if abs(cur_loss - prev_loss) < 0.1:
                print('current loss - prev loss < 0.1, breaking')
                break
            prev_loss = train_loss
```



Experimental Results from Our Implementation

[tom to insert]



########## END TOM WORKING ON THIS SECTION ##########

Experimental Results from Paper

The authors tested the visualization performance of VASC together with four state-of-the-art dimension reduction methods: PCA, ZIFA, t-SNE, and SIMLR. They used 20 datasets with different number of cells included and sequencing protocols used. Performance assessments were measured using k-means clustering. 



The top panel (Figure 3A) shows the NMI and ARI values for each method on each dataset. NMI stands for normalized mutual information, which is calculated as $$NMI(P, T) = \frac{MI(P, T)}{\sqrt{H(P) H(T)}}$$, where $$P$$ is the predicted clustering results, and $$T$$ is the known cell types, and $$H(P)$$ and $$H(T)$$ are the entropy of $$P$$ and $$T$$,  the mutual information between them as $$MI(P,T)$$.

ARI stands for adjusted rand index, it is calculated as 

$$\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] \Big/ \binom{n}{2}}
{\frac{1}{2} \left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] \Big/ \binom{n}{2}}$$

where $$n$$ is the total number of samples, $$a_i$$ is the number of samples appearing in the $$i$$-th cluster of $$P$$, $$b_j$$ is the number of samples appearing in the $$j$$-th types of $$T$$, and $$n_{ij}$$ is the number of overlaps between the $$i$$-th cluster of $$P$$ and the $$j$$-th type and $$T$$. 

From the results from the top panel, we see that VASC outperformed the other methods in terms of NMI and ARI in most cases (best performances achieved on 15 and 17 out of the 20 datasets, respectively).

The lower panel shows the statistics of the ranks of the compared methods based on NMI and ARI values. For each dataset, NMI and ARI values given by different algorithms were ranked in descending order, with rank 1 indicative of the highest NMI or ARI values. The number of ranks achieved by these algorithms in the 20 datasets is then counted for distribution. As we can see from the results, VASC always ranked in the top two methods of all the tested datasets in terms of NMI and ARI.

Conclusions
Overall, the results from the paper suggested that VASC has broad compatibility with various kinds of scRNA-seq datasets and performs better than PCA and ZIFA, especially when the sample sizes are larger. VASC achieves superior performance in most cases and is broadly suitable for different datasets with different data structures in the original space.






# References
1. Deep Learning (Ian J. Goodfellow, Yoshua Bengio and Aaron Courville), MIT Press, 2016.
2. Kingma, D. P., & Welling, M. (2019). An Introduction to Variational Autoencoders. ArXiv. https://doi.org/10.1561/2200000056
3. Wang, Dongfang, and Jin Gu. "VASC: dimension reduction and visualization of single-cell RNA-seq data by deep variational autoencoder." Genomics, Proteomics and Bioinformatics 16.5 (2018): 320-331.
4. Li, X., Wang, CY. From bulk, single-cell to spatial RNA sequencing. Int J Oral Sci 13, 36 (2021). https://doi.org/10.1038/s41368-021-00146-0
5. Yu X, Abbas-Aghababazadeh F, Chen YA, Fridley BL. Statistical and Bioinformatics Analysis of Data from Bulk and Single-Cell RNA Sequencing Experiments. Methods Mol Biol. 2021;2194:143-175. doi: 10.1007/978-1-0716-0849-4_9. PMID: 32926366; PMCID: PMC7771369.
6. Srivastava, Nitish, et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." Journal of Machine Learning Research, vol. 15, no. 56, 2014, pp. 1929-1958, http://jmlr.org/papers/v15/srivastava14a.html.



-------------------------------------------------------------

![image tooltip here](/assets/image.jpg)
