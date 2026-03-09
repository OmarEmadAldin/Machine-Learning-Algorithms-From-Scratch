import numpy as np

class NaiveBayes:
    def fit(self, x , y ):
        n_samples , n_features = x.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        # Calculate mean , var and prior for each class
        self._mean = np.zeros((n_classes,n_features),dtype=np.float64)
        print ("the mean matrix is " , self._mean)
        self._var = np.zeros((n_classes,n_features),dtype=np.float64)
        self._prior = np.zeros(n_classes , dtype=np.float64)

        for idx , c in enumerate(self._classes):
            X_c = x[ y==c ]
            print(X_c)
            self._mean[idx,:] = X_c.mean(axis=0)
            self._var[idx,:] = X_c.var(axis=0)
            self._prior[idx] = X_c.shape[0]/float(n_samples)

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self,X):
        posteriors =[]
        for idx , c in enumerate (self._classes):
            prior = np.log(self._prior[idx])
            posterior = np.sum(np.log(self._pdf(idx,X)))
            posterior = posterior +prior
            posteriors.append(posterior)
        
        output = self._classes[np.argmax(posteriors)]
        return output
    
    def _pdf(self ,class_idx ,x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        numer = np.exp((-(x-mean)**2) / (2*var))
        denom = np.sqrt(2*np.pi *var)

        pdf = numer / denom
        return pdf