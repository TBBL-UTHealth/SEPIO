# MDPO_GUI Walkthrough

How to use MDPO_GUI tab-by-tab.


## 1. Project Setup
- Project Metadata
    - Name: Label your project file
    - Data Folder: Choose where your project file lives
- Brain Surface
    - Provide the whole-brain .mat file.
- ROI Surfaces
    - Provide the ROI .mat files to define the target tissue.
- Structural Avoidance
    - Provide .mat files with defined spheres of avoidance.
    - Built for future development into vasculature and regional avoidance.
    - Demo mode provides an example sphere to avoid at the center of each ROI.
- Lead Field Files
    - Provide .npz files to use as defined devices in this space.
    - Each lead field added generates a new column below to provide device details.
- Device Configuration
    - Provide the details relevant to each provided device
    - See device_configs.md for details on the provided lead fields in the Zenodo database.
    - Enable/disable device proximity and depth limitations.

## 2. Optimization
- Optimization Controls
    - Select the optimization method (Minibatch SGD, Genetic, Anneal, Multi-anneal, Brute Force).
    - Set the measurement to optimize on (IC, SNR, or Voltage; IC by default).
- Parameters
    - Set specific values for the given optimization method. Tool tips are provided.
- Start
    - Run the optimization and monitor status, progress, and best fitness.
- Remove optimization results
    - Selectively or entirely remove optimization results to clean a project file.

## 3. Assessment
- Select Optimization Result
    - Refresh the list and choose the optimization to assess.
    - This process only saves and assesses the best solution from optimization.
- Fitness Plot
    - Plot the fitness curve over epochs; save the figure if desired.
- Visualize Best
    - Generate an interactable 3D visualization of the placement results in the brain.
    - "Only ROI" reduces the visuals to only show the target tissue.
    - "Show Avoidance" represents avoided regions with a wire frame sphere.
    - "Debug" shows relevant vectors for reference.
    - "X-Ray View" shows the same model such that the surface visibility is inverted.
- Exports
    - Export best solution and per-epoch summaries to CSV.

## 4. SEPIO
- Select Result
    - Choose an optimization result to analyze in SEPIO.
- Voltage Computation
    - Re-compute voltage arrays over the ROI using the selected device configuration.
    - Voltage distribution histogram is available.
    - See the provided output in "Voltage Summary".
- SEPIO Monte-Carlo
    - Run classification trials across sensors/devices; choose ranges and iterations.
    - Tool tips are provided
    - Compute spatial is a work in progress.
    - Run SEPIO (Progress bar currently non-functional)
- SEPIO Results Summary
    - View summary metrics and plot sensor/device accuracy.
- Select SEPIO Result
    - Offers a manner of removing one or all SEPIO results to clean up a project file.

## 5. Plotting
- Datasets
    - Add SEPIO results to the dataset list; remove selected entries as needed.
    - These datasets will be plotted together in the following graphics.
- Accuracy Line Plots
    - Plot accuracy trends by sensor count for each SEPIO dataset.
- Accuracy Heatmap
    - Plot a sensors-vs-devices grid to compare accuracy by sensor and device count.
    - This heatmap relies on the provided SEPIO datasets containing different numbers of devices.
    - It is best to compare the same optimization and SEPIO configuration with only varying the device type/count.

## 6. Tips
- If many optimizations or SEPIO trials have been run, consider starting a new project or clearing old data. This is primarily relevant to larger datasets.
