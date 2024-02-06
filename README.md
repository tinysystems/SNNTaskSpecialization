# SNNTaskSpecialization

## Code reference: 
* This is the code of the work published in MA Lebdeh, et al. "Efficient Processing of Spiking Neural Networks via Task Specialization", IEEE TETCI, 2024.  
* The training of the SNNs are adopted fron tdBN from https://github.com/thiswinex/STBP-simple  


## Notes:
* The codes are self contained and should run when following the three steps below. 
* In the main file "example_cifar10_vgg9.py", make sure you change directories of the dataset and the location where you want to save the model.


## Steps to train a sub-SNN: 

1. Insall all the packages in the file "requirments.txt" into your virtual environment.

2. Run your virtual environment.

3. Run the python file for training
   * python "example_cifar10_vgg9.py" --scale-factor 8 --epochs 260  --reset-model 1  --subSNN-index 0
   * Tune the hyperparameters by passing the parser's parameters as shown in the example above.



## Steps to train the aggregation function: 
* Note: You may create your own code to train the aggregation function. Make sure all directories are changed to some locations in your machine. 

1. Run the virtual environment. 
2. In the file "preprocessed_outputs_dataset_generator_cifar10_vgg9.py", set the variable 'create_outputs' to 1. Then, run "preprocessed_outputs_dataset_generator_cifar10_vgg9.py". 
3. In the file "preprocessed_outputs_dataset_generator_cifar10_vgg9.py", set the variable 'create_outputs' to 0. Then, run "preprocessed_outputs_dataset_generator_cifar10_vgg9.py". This creates a dataset to train the aggregation function.
4. Train the aggregation function by running the file "aggregation_function_training_cifar10_vgg9.py"
