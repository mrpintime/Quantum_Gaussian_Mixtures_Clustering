import argparse
import numpy as np
import cv2
from PIL import Image
import random
import warnings
warnings.filterwarnings('ignore')

# define QuantumGaussianMixture Model 
class QuantumGaussianMixture:
    def __init__(self, n_components, compare=False, n_iter=100, tol=1e-6):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.compare = compare 

    def initialize_parameters(self, X):
        n_samples, n_features = X.shape

        # K-means++ initialization for means
        self.mu = np.empty((self.n_components, n_features))
        self.mu[0] = X[np.random.choice(n_samples)]
        
        for k in range(1, self.n_components):
            distances = np.min([np.sum((X - self.mu[j]) ** 2, axis=1) for j in range(k)], axis=0)
            probs = distances / np.sum(distances)
            self.mu[k] = X[np.random.choice(n_samples, p=probs)]
        
        # Initialize alpha uniformly
        self.alpha = np.ones(self.n_components) / self.n_components
        
        # Initialize covariance matrices to the covariance of the dataset
        self.C = np.array([np.cov(X.T) for _ in range(self.n_components)])
        
        # Initialize phases uniformly between 0 and 2*pi
        self.phi = np.random.uniform(0, 2 * np.pi, self.n_components)

    def wave_function(self, x, mu, C, phi):
        d = len(x)
        if d == 1: # If dimension of dataset was 1 it means we have 1 feature and then we do not have covariance matrix instead we have scaler variance 
            C_inv = C ** -1
            Z_k = (2 * np.pi)**(d / 2) * C**0.5
        elif d > 1:
            C_inv = np.linalg.inv(C)
            Z_k = (2 * np.pi)**(d / 2) * np.linalg.det(C)**0.5
            
        diff = x - mu
        exp_part = np.exp(-0.25 * np.dot(diff.T, np.dot(C_inv, diff)))
        phase_part = np.exp(-1j * phi)
        return exp_part * phase_part / Z_k

    def probability(self, x, mu, C):
        d = len(x)
        if d == 1:
            C_inv = C ** -1
            Z_k = (2 * np.pi)**(d / 2) * C**0.5
        elif d > 1:
            C_inv = np.linalg.inv(C)
            Z_k = (2 * np.pi)**(d / 2) * np.linalg.det(C)**0.5
            
        diff = x - mu
        exp_part = np.exp(-0.5 * np.dot(diff.T, np.dot(C_inv, diff)))
        return exp_part / Z_k

    def mixture_wave_function(self, x):
        psi = sum(self.alpha[k] * self.wave_function(x, self.mu[k], self.C[k], self.phi[k])
                  for k in range(self.n_components))
        return psi

    def mixture_probability(self, x):
        psi = self.mixture_wave_function(x)
        return np.abs(psi)**2

    def e_step(self, X):
        n_samples = X.shape[0]
        Q = np.zeros((n_samples, self.n_components))

        for i in range(n_samples):
            total_prob = self.mixture_probability(X[i])
            for k in range(self.n_components):
                Q[i, k] = self.alpha[k] * self.probability(X[i], self.mu[k], self.C[k])
            Q[i, :] /= total_prob
        return Q

    def m_step(self, X, Q):
        n_samples, n_features = X.shape

        for k in range(self.n_components):
            N_k = np.sum(Q[:, k], axis=0)
            self.alpha[k] = N_k / n_samples
            self.mu[k] = np.sum(X * Q[:, k, np.newaxis], axis=0) / N_k
            diff = X - self.mu[k]
            self.C[k] = np.dot((Q[:, k, np.newaxis] * diff).T, diff) / N_k
            
            self.phi[k] = np.angle(np.sum(Q[:, k] * np.exp(1j * self.phi[k])))

    def fit(self, X):
        if not self.compare:
            self.initialize_parameters(X)

        for iteration in range(self.n_iter):
            Q = self.e_step(X)
            prev_mu = self.mu.copy()
            self.m_step(X, Q)

            if np.linalg.norm(self.mu - prev_mu) < self.tol:
                break

    def predict(self, X):
        Q = self.e_step(X)
        return np.argmax(Q, axis=1)

    def predict_proba(self, X):
        return self.e_step(X)
# define a function to add noise to image
# Function references = https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
    

def main(image_path, output_file, n_components, n_iter):
    # Your main script logic here
    print(f"Input File: {image_path}")
    print(f"Output File: {output_file}")
    print(f"Number of Cluster: {n_components}")
    print(f"Number of Iterration: {n_iter}")
    
    image = cv2.imread(image_path)
    # some preprocessing for less computation
    resized_image = cv2.resize(image, (300, 300)) 
    noise_img = sp_noise(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB),0.05)
    gray_image = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)
    # Reshape the image to a 2D array of pixels
    # Each row is a pixel value
    pixel_values = gray_image.reshape((-1, 1))
    # Convert to float 
    pixel_values = np.float64(pixel_values)
    
    # define QuantumGaussianMixture Model
    qgm = QuantumGaussianMixture(n_components=n_components, compare=False, n_iter=n_iter, tol=1e-6)
    # Fit the Quantum Gaussian Mixture model
    qgm.fit(pixel_values)
    
    # Prediction Step
    # Labels of the Quantum Gaussian Mixture model
    labels = qgm.predict(pixel_values)
    # reshape to original image
    image_form_segmentated = labels.reshape(gray_image.shape)
    # Normalize the data
    min_val = np.min(image_form_segmentated)
    max_val = np.max(image_form_segmentated)

    # Scale to 0-255 range
    image_form_segmentated = 255 * (image_form_segmentated - min_val) / (max_val - min_val)
    image_form_segmentated = image_form_segmentated.astype(np.uint8)
    im = Image.fromarray(image_form_segmentated)
    im.save(f"{output_file}")
    
    print('Thanks for using my script, your image Saved successfully.')
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input Image File')
    parser.add_argument('--components', type=int, required=True, help='Number of Clusters')
    parser.add_argument('--iter', type=int, required=True, help='Number of Iterations')
    parser.add_argument('--output', type=str, required=True, help='Path to the output Image File')

    args = parser.parse_args()
    print(args)

    main(args.input, args.output, args.components, args.iter)
