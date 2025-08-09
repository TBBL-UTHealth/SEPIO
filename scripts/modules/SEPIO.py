# Libraries
import warnings, math
import numpy as np
import pysensors as ps
from pysensors.classification import SSPOC
from sklearn import metrics

class sepio:
    """
    SSPOC training, classification, and sensor sorting with Monte-Carlo and support
    functions for crossed datasets and reduced sensor groups.
    """

    def mc_train(X,y,Xt,yt,sensor_div,MCcount,noise,replicates,nbasis,spatial,l1,rep_subdiv):
        """
        Performs Monte-Carlo style SEPIO training, testing, and sensor sorting.
        
        This function implements the SEPIO (Sparse Sensor Placement for Intracranial Optimization)
        algorithm which uses SSPOC (Sparse Sensor Placement Optimization for Classification) to
        find optimal electrode placements for neural signal classification.
        
        Parameters:
        -----------
        X : array_like, shape (sources, sensors)
            Training data containing signal measurements from different sources/dipoles
        y : array_like, optional
            Class labels for training data X. If not provided, labels are auto-generated
        Xt : array_like, optional
            Test data for cross-validation, used for evaluating different datasets
        yt : array_like, optional  
            Test labels corresponding to Xt
        sensor_div : int
            Sensor increment for iterative testing (defines granularity of sensor range)
        MCcount : int
            Number of Monte-Carlo iterations to perform for robust statistics
        noise : float
            Standard deviation of Gaussian noise added to simulate realistic conditions
        replicates : int
            Number of data replications for augmentation (if single samples per class)
        nbasis : int, optional
            Number of basis functions for dimensionality reduction
        spatial : bool, optional
            Whether to return spatial/source-space classification accuracies
        l1 : float, optional
            L1 regularization parameter for SSPOC
        rep_subdiv : int, optional
            Number of subdivisions for replicate testing
        
        Returns:
        --------
        MCcoefs : ndarray
            Sensor coefficients indicating electrode importance
        MCaccs : ndarray  
            Classification accuracies in sensor space across different sensor counts
        MCsaccs : ndarray, optional
            Classification accuracies in source space (if spatial=True)
        sensor_range : ndarray
            Array of sensor counts tested
        """
        # Maximum sensors for the dataset
        NUM_ELECTRODES = X.shape[-1]

        # Handle labels and replicates
        if y is not None:
            _,counts = np.unique(y,return_counts=True)
            counts = np.max(counts)
        else: # Produce labels if needed
            counts = 1
            y = np.arange(X.shape[0]) # labels
        # Separate test set labels
        if (Xt is not None) and (yt is None):
            yt = np.arange(Xt.shape[0]) # labels
            countst = 1
        elif (Xt is not None) and (yt is not None):
            _,countst = np.unique(yt,return_counts=True)
            countst = np.max(countst)
        else:
            countst = None

        # Divide sensor testing if not given
        if sensor_div is None:
            sensor_div = NUM_ELECTRODES//2
            print("No sensor_div provided, set to ",sensor_div)
            
        # Internal settings & defaults
        if rep_subdiv is None:
            rep_subdiv = replicates//4 # Number of replicate subdivisions
        if l1 is None:
            l1 = 0.001
        if spatial is None:
            spatial = False
        
        # Define basis mode selection; SVD, Identity, or RandomProjection
        if nbasis > NUM_ELECTRODES:
            nbasis = NUM_ELECTRODES
        basis = ps.basis.SVD(n_basis_modes=int(nbasis))

        # NaN to zero
        X = np.nan_to_num(X)
        if Xt is not None:
            Xt = np.nan_to_num(Xt)
        
        #--- Generate replicates or process provided replicates ---#
        # Test data
        if countst == 1: # Separate test data; one sample per label
            X_test = np.repeat(Xt,rep_subdiv,axis=0)
            y_test = np.repeat(yt,rep_subdiv,axis=-1)
        elif countst is not None: # Separate test data; more than one sample per label
            # Does not create replicates for multiple provided samples
            X_test = Xt
            y_test = yt
        elif countst is None: # Use training data in same replicate
            # Done with training data to mimic what is given for X and y
            pass
        
        # Training data
        if counts == 1: # One sample given per label
            if countst is None: # Same test data; subdividing reps for testing
                X_test = np.repeat(X,rep_subdiv,axis=0)
                y_test = np.repeat(y,rep_subdiv,axis=-1)
            y = np.repeat(y,replicates,axis=-1)
            X = np.repeat(X,replicates,axis=0)
            
        else: # Multiple samples given
            # Pass, replicates are not made
            y = y
            X = X
            if countst is None: # Same test data; no replicates
                X_test = np.copy(X)
                y_test = np.copy(y)
        #--- Initialize outputs ---#
        # Coefficients for sensor sorting
        if np.unique(y).shape[0] <= 2: # Only 2 labels -> (sensors,)
            MCcoefs = np.zeros((NUM_ELECTRODES,))
        else: # More than 2 labels -> (sensors,labels)
            MCcoefs = np.zeros((NUM_ELECTRODES,np.unique(y).shape[0]))
        # Sensor range to iterate over for test classification; slows down significantly with smaller divisions
        sensor_range = np.arange(sensor_div,NUM_ELECTRODES+1,sensor_div)
        # Accuracy in sensor space
        MCaccs = np.zeros((sensor_range.shape[0]))
        # Accuracy in source space
        MCSaccs = np.zeros((np.unique(y).shape[0]))
        # Make stable labels list for MC cycles
        y_set = np.copy(y)
        #--- Iterate Monte-Carlo ---#
        for i in range(MCcount):
            print("Starting run",i+1,"of",(MCcount))
            
            #--- Train the model ---#
            print("Training model...")
            warnings.simplefilter("ignore")
            model = None # Clear model from prior Monte-Carlo
            model = SSPOC(l1_penalty=l1,basis=basis)
            # Overlay noise, shuffle, and train
            X_train = np.copy(X) + np.random.normal(scale=noise,size=X.shape)
            shuf = np.random.permutation(np.array(range(len(y))))
            y = y_set.copy()
            y = y[shuf]
            X_train = X_train[shuf]
            model.fit(X_train, y)
            #del X_train # Clear memory space
            # Save coefficients
            MCcoefs += model.sensor_coef_

            #--- Test the model ---#
            print("Testing model (SENSORS)...")
            accs = np.zeros(MCaccs.shape)
            k = 0
            saccs = np.zeros(MCSaccs.shape)
            n = 0
            for i,s in enumerate(sensor_range):
                print(f"Testing {s} sensors of {sensor_range}")
                # Standard dataset sensor selection
                model.update_sensors(n_sensors=s, xy=(X_train, y), quiet=True)
                sensors = model.selected_sensors
                # Create replicates and test in sets of 2x labels IF replicates are used
                if ((counts == 1) and (countst is None)) or (countst == 1): # replicates used
                    for k in range(replicates//rep_subdiv):
                        # Create replicate
                        X_testk = np.copy(X_test) + np.random.normal(scale=noise,size=X_test.shape)
                        # Shuffle
                        shuf = np.random.permutation(np.array(range(len(y_test))))
                        X_testk = X_testk[shuf]
                        y_testk = np.copy(y_test)[shuf]
                        # Test and record scores
                        y_pred = model.predict(X_testk[:,sensors])
                        acc = metrics.accuracy_score(y_testk,y_pred)
                        print(f"Sensor accuracy: {acc}")
                        accs[i] += acc
                        del X_testk # Clear memory
                else: # No replicates in test set (Simulated)
                    subdivs = countst//rep_subdiv # Cut down to only whole sections of length 'rep_subdiv'
                    print(f"Using {subdivs} samples at a time, totalling {subdivs*rep_subdiv} of possible {countst}.")
                    # Shuffle
                    shuf = np.random.permutation(range(len(y_test)))
                    X_testk = np.copy(X_test)[shuf]
                    y_testk = np.copy(y_test)[shuf]
                    for k in range(subdivs):
                        # Test and record scores
                        y_pred = model.predict(X_testk[rep_subdiv*k:rep_subdiv*(k+1),sensors])
                        acc = metrics.accuracy_score(y_testk,y_pred)
                        print(f"Sensor accuracy: {acc}")
                        accs[i] += acc
                    #k = 0 # Set divisions to 0 if not subdividing
            
            
            if spatial: # Avoid calc if unneeded as it takes a LONG time
                # Accuracy spatial per label
                print("Testing model (SOURCES)...")
                # Model should still be loaded with the total sensor count
                #model.update_sensors(n_sensors=NUM_ELECTRODES, xy=(X_train, y), quiet=True)
                if ((counts == 1) and (countst is None)) or (countst == 1): # replicates used
                    for n in range(replicates//rep_subdiv):
                        # Create replicate
                        X_testk = np.copy(X_test) + np.random.normal(scale=noise,size=X_test.shape)
                        # Shuffle
                        shuf = np.random.permutation(np.array(range(len(y_test))))
                        inv_shuf = np.argsort(shuf)
                        X_testk = X_testk
                        y_testk = np.copy(y_test)
                        # Test with all sensors and record scores
                        y_pred = model.predict(X_testk[:,sensors])
                        for j in range(saccs.shape[0]): # j for each data labels
                            j_index = np.where(y_testk==j)[0]
                            saccs[j] += metrics.accuracy_score(y_testk[j_index],y_pred[j_index])
                    print(f"Spatial accuracy: {np.mean(saccs)/(n+1)}")
                else: # No replicates for test set
                    pass
            
            # Save values each Monte-Carlo cycle; divide for each replicate sums
            MCaccs += accs/(k+1)
            MCSaccs += saccs/(n+1)
        
        # Divide for sum of Monte-Carlo cycles
        print("Finishing up..")
        MCcoefs /= MCcount
        MCaccs /= MCcount
        MCSaccs /= MCcount

        print("Finished!")
        # Ouputs coefficients, test accuracy (sensor space),
        #   test accuracy (source space), and sensor range array
        return MCcoefs, MCaccs, MCSaccs, sensor_range