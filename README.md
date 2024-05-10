<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script> 
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>


Math behind VAE
Now we understand the basic structure of VAE, let's take a deeper dive into the math.
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


-------------------------------------------------------------
![image tooltip here](/assets/image.jpg)
