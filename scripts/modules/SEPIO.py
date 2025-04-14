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
        Performs Monte-Carlo style SEPIO training, testing, and sorting.
        
        Inputs:
        `X` (float array) is the train data in (sources,sensors) format; also generates labels.
        (Opt.) `y` (int) is the matching labels for `X`. Not necessary if X only contains one
            sample for each label. If provided, replicates are not made (must be made manually).
        (Opt.) `Xt` (float array) is the test data, only defined for crossed datasets.
        (Opt.) `yt` (int) same as `y`, but for `Xt`.
        `sensor_div` (int) is the number of sensors to iterate at a time for test processing.
        `MCcount` (int) is the number of total cycles to perform.
        `noise` (float) is the standard deviation of a gaussian noise profile overlaid on datasets.
        `replicates` (int) is the number of dataset duplication to perform for training and testing.
            If multiple samples are provided for each label, only those are used, split for train/test.
        (Opt.) `nbasis` (int) is the number of basis modes permitted.
        (Opt.) `spatial` (bool) is an option to return the classification accuracy per label.
        (Opt.) `l1` (float) allows the changing of l1 penalty for optimization.
        rep_subdiv (int) is the number of replicates tested at a time. Lower values use less RAM and take
            slightly longer, but very low values (~<6) may degrade results. Defaults to replicates//4.
        
        Returns: (Coefficients, sensor/input accuracy, spatial/label accuracy, sensor_list)
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
            y = y[shuf]
            X_train = X_train[shuf]
            model.fit(X_train, y)
            #del X_train # Clear memory space
            # Save coefficients
            MCcoefs += model.sensor_coef_

            #--- Test the model ---#
            print("Testing model (sensors)...")
            accs = np.zeros(MCaccs.shape)
            k = 0
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
                        accs[i] += acc
                        del X_testk # Clear memory
                else: # No replicates in test set (Simulated)
                    subdivs = countst//rep_subdiv # Cut down to only whole sections of length 'rep_subdiv'
                    print(f"Using {subdivs} samples at a time, totalling {subdivs*rep_subdiv} of possible {countst}.")
                    # Shuffle
                    shuf = np.random.permutation(range(len(y_test)))
                    X_test = X_test[shuf]
                    y_test = y_test[shuf]
                    for k in range(subdivs):
                        # Test and record scores
                        y_pred = model.predict(X_test[rep_subdiv*k:rep_subdiv*(k+1),sensors])
                        acc = metrics.accuracy_score(y_test,y_pred)
                        accs[i] += acc
                    #k = 0 # Set divisions to 0 if not subdividing
            
            saccs = np.zeros(MCSaccs.shape)
            n = 0
            if spatial: # Avoid calc if unneeded as it takes a LONG time
                # Accuracy spatial per label
                print("Testing model (sources)...")
                model.update_sensors(n_sensors=NUM_ELECTRODES, xy=(X_train, y), quiet=True)
                if ((counts == 1) and (countst is None)) or (countst == 1): # replicates used
                    for n in range(replicates//rep_subdiv):
                        # Create replicate
                        X_testk = np.copy(X_test) + np.random.normal(scale=noise,size=X_test.shape)
                        # Shuffle
                        shuf = np.random.permutation(np.array(range(len(y_test))))
                        X_testk = X_testk[shuf]
                        y_test = y_test[shuf]
                        # Test with all sensors and record scores
                        y_pred = model.predict(X_testk)
                        for j in range(y_test.shape[0]): # j for each data labels
                            j_label = np.where(y_test==j)[0]
                            saccs[j] += metrics.accuracy_score(y_test[j_label],y_pred[j_label])
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