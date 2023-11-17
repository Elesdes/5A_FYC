# Text2PythonCode (VAEGAN) | FYC 2023-2024

## Exercise Steps

### Environment setup
- [Install Tensorflow](https://www.tensorflow.org/install/pip)
- Install additional libraries : `pip install -r requirements.txt`

### Download the dataset
Download the dataset from [here](https://www.kaggle.com/datasets/linkanjarad/coding-problems-and-solution-python-code/data?select=ProblemSolutionPythonV3.csv)

### Data Preparation
- Using ***pandas*** library, we will read the dataset and extract the ***Problem*** and ***Python Code*** columns. 
- Then, we will clean the data by removing the non-ascii characters and the rows that have empty values. 
- Finally, we will save the cleaned data in a new CSV file named `clean_dataset.csv`.
- Finally, we will split the dataset into train and test sets and save them respectively as `train_dataset.csv` and `test_dataset.csv`.

### Tokenization & Preprocessing
Using `tensorflow.keras.preprocessing.text.Tokenizer`, we will tokenize the ***Problem*** and ***Python Code*** columns.

### Build the Variational Autoencoder (VAE)
- Define the VAE architecture using the Keras API in TensorFlow. 
- Create the encoder and decoder networks using the Functional API. 
- Then, create the VAE model by subclassing the `tensorflow.keras.Model` class. 
- Use the ***Mean Squared Error (MSE)*** as the reconstruction loss.

### Train the VAE
- Compile and train the VAE on your dataset.
- Monitor the training progress and visualize the reconstructed outputs.

### Build the Generative Adversarial Network (GAN)
- Define the GAN architecture using the Keras API in TensorFlow.
- Use the ***Binary Crossentropy*** as the loss function.

### Train the GAN
- Compile and train the GAN on your dataset.
- Use the latent space representation obtained from the VAE as input to the GAN's generator.
- Train the discriminator to distinguish between real and generated code snippets.

### Combine VAE & GAN (VAEGAN) for Text-to-Code Generation
Combine the trained encoder from the VAE with the generator from the GAN to create a text-to-code generator.\
***Note : This generator should take a textual description as input and produce a code snippet as output.***

### Training End-to-End
- Train the entire model end-to-end.
- Optimize both the VAE and GAN hyperparameters to improve the performance of the model.
- Use a loss function that balances the reconstruction loss and the adversarial loss from the GAN.

### Evaluate & Experiment
- Evaluate the model on the test dataset.
- Experiment with different hyperparameters and architectures to improve the performance of the model.

### Additional Tips :
- Experiment with different activation functions, layer sizes, and learning rates in addition to experimenting with number of neurons per layer.
- To optimize the hyperparameters in a more autonomous way, using for example ***Bayesian Optimization*** algorithm, you can use [KerasTuner](https://keras.io/keras_tuner/).