import numpy as np
import pickle

class WiSARD:
    """
    WiSARD: Weightless Neural Network.
    RAM-based learning for extreme edge efficiency.
    """
    
    def __init__(self, num_bits: int = 16, bleach: int = 1):
        self.num_bits = num_bits # Address size (tuple size)
        self.bleach = bleach # Filtering threshold
        self.discriminators = {} # Class -> RAM set
        self.mapping = None # Global input -> RAM mapping
        self.classes_ = None
    
    def fit(self, X, y):
        # Input X expected to be binary or we binarize it.
        # For simplicity, assume X is already flattened binary (0/1) or float 0..1
        
        if self.mapping is None:
            n_inputs = X.shape[1]
            # Create random mapping: which inputs go to which RAM address bit
            # We need n_rams = n_inputs / num_bits
            self.mapping = np.random.permutation(n_inputs)
        
        # Adaptive thresholding: use mean per sample or global mean?
        # Per-sample mean helps if illumination/intensity varies. 
        # Deep features (ReLU) are non-negative.
        # Let's use per-sample mean to get roughly 50% sparcity.
        X_bin = (X > X.mean(axis=1, keepdims=True)).astype(int)
        
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            if cls not in self.discriminators:
                self.discriminators[cls] = {} # Dict of RAMs? 
                # Actually, effectively a list of sets/counters.
            
            X_cls = X_bin[y == cls]
            for sample in X_cls:
                self._train_sample(sample, self.discriminators[cls])
                
        return self

    def _train_sample(self, sample, discriminator):
        # Apply mapping
        mapped = sample[self.mapping]
        # Split into chunks (tuples) of size num_bits
        # And write to RAM
        # Efficient way: view as chunks, convert binary chunk to int address.
        
        # Pad if needed
        cutoff = (len(mapped) // self.num_bits) * self.num_bits
        chunks = mapped[:cutoff].reshape(-1, self.num_bits)
        
        # Convert binary row to integer
        # e.g. [1, 0, 1] -> 5
        powers = 1 << np.arange(self.num_bits)[::-1]
        addresses = chunks @ powers
        
        # Write
        for rams_idx, addr in enumerate(addresses):
            if rams_idx not in discriminator:
                discriminator[rams_idx] = {} # Use dict for sparse RAM
            # Increment counter or just set bit? Standard WiSARD is just set bit.
            # DRASiW uses counters. Let's use simple set bit (dict key exists).
            discriminator[rams_idx][addr] = 1

    def predict(self, X):
        if self.classes_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        # Same adaptive thresholding
        X_bin = (X > X.mean(axis=1, keepdims=True)).astype(int)
        scores = []
        
        classes = list(self.classes_)
        
        for sample in X_bin:
            sample_scores = []
            
            mapped = sample[self.mapping]
            cutoff = (len(mapped) // self.num_bits) * self.num_bits
            chunks = mapped[:cutoff].reshape(-1, self.num_bits)
            powers = 1 << np.arange(self.num_bits)[::-1]
            addresses = chunks @ powers
            
            for cls in classes:
                score = 0
                disc = self.discriminators[cls]
                for rams_idx, addr in enumerate(addresses):
                    if rams_idx in disc and addr in disc[rams_idx]:
                        score += 1
                sample_scores.append(score)
            
            # Winner takes all
            best_cls_idx = np.argmax(sample_scores)
            scores.append(classes[best_cls_idx])
            
        return np.array(scores)
