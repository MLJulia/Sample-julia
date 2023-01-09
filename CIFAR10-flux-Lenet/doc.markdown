LeNet-5 is a convolutional neural network (CNN) that was introduced by Yann LeCun et al. in their 1998 paper, "Gradient-Based Learning Applied to Document Recognition." It was one of the first successful applications of CNNs on a large-scale image recognition task, and it is still widely used today as a starting point for many image recognition tasks.
![picture 1](../images/b5e5e3adbc2b907cb850ccb51d3c79766106b6848746e9b153e7ce31107b5ba4.png)  
[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, november 1998.

In this tutorial, we will see how to train a LeNet-5 model on the CIFAR-10 dataset using Flux.jl, a machine learning library for the Julia programming language.
Installing Flux.jl

<!-- In this post we are going to train CIFAR-10 with LeNet-5 and Flux.jl -->

To train LeNet-5 on CIFAR-10 with Flux.jl, you will need to have the following installed:

    Julia 1.5 or higher
    Flux.jl
    CIFAR-10 dataset

<!-- LeNet-5 is a convolutional neural network (CNN) designed by Yann LeCun and his colleagues in the 1990s. It was one of the first successful CNNs and is widely used as a benchmark for comparing the performance of different CNN architectures on image classification tasks. In this tutorial, we will train LeNet-5 on the CIFAR-10 dataset using Flux.jl, which is a machine learning library for the Julia programming language. -->


If you don't have Julia and Flux.jl installed, you can follow the instructions on the Julia website (https://julialang.org/downloads/) and the Flux.jl documentation (https://fluxml.ai/getting_started/) to install them. 

First, we need to install Flux.jl. If you don't already have Julia installed on your system, you can download it from https://julialang.org/downloads/. Once you have Julia installed, open the Julia REPL (read-eval-print-loop) by running julia in your terminal.

Next, we will install Flux.jl using the Pkg package manager. In the Julia REPL, type the following:
```julia
using Pkg
Pkg.add("Flux")
```

<!-- I personally use VsCode for IDE but you may use any editor you find useful. -->




We need to import the following packages.

```julia 
using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle, @epochs
using Base.Iterators: repeated, partition
using MLDatasets

```

If you noticed you getting an error saying some packages doesn't exist, you can install them as:
```julia
import Pkg; Pkg.add("Name-of-package") # replace the name-of-package with your package 
```


## Step 1: Loading the Data and Preprocessing the Data


Next, we need to download and load the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The 10 classes are:

    airplane
    automobile
    bird
    cat
    deer
    dog
    frog
    horse
    ship
    truck

We will use the MLDatasets package to do this. This package includes functions to download and load the dataset.



```julia
# load CIFAR-10 training set
trainX, trainY = CIFAR10.traindata()
testX,  testY  = CIFAR10.testdata()
```
Note that the very first time you do this, it may take a long time or even ask you
```julia

# Do you want to download the dataset from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz to "C:\Users\Farhad\.julia\datadeps\CIFAR10"?
# [y/n]

```
You need to type y, then enter. 

One good practice is to write a function to load and clean the data:
```julia

function get_data(batchsize; dataset = MLDatasets.CIFAR10, idxs = nothing, T= Float32)
    """
    idxs=nothing gives the full dataset, otherwise (for testing purposes) only the 1:idxs elements of the train set are given.
    dataset is the datasets we will use

    """
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" 

    # Loading Dataset
    if idxs===nothing
        xtrain, ytrain = dataset(Tx=T,:train)[:]
        xtest, ytest = dataset(Tx=T,:test)[:]
	else
        xtrain, ytrain = dataset(Tx=T,:train)[1:idxs]
        xtest, ytest = dataset(Tx=T, :test)[1:Int(idxs/10)]
    end

    # Reshape Data to comply to Julia's (width, height, channels, batch_size) convention in case there are only 1 channel (eg MNIST)
    if ndims(xtrain)==3
        w = size(xtrain)[1]
        xtrain = reshape(xtrain, (w,w,1,:))
        xtest = reshape(xtest, (w,w,1,:))
    end
    
    # construct one-hot vectors from labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=batchsize)

    return train_loader, test_loader
end
```

The function `get_data` performs the following tasks:

* **Loads some dataset:** Loads the train and test set tensors. Here we set the defualt to CIFAR-10
* **Reshapes the train and test data:**  Notice that we reshape the data so that we can pass it as arguments for the input layer of the model.
* **One-hot encodes the train and test labels:** Creates a batch of one-hot vectors so we can pass the labels of the data as arguments for the loss function. For this example, we use the [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) function and it expects data to be one-hot encoded. 
* **Creates mini-batches of data:** Creates two DataLoader objects (train and test) that handle data mini-batches of size defined by minibatch. We create these two objects so that we can pass the entire data set through the loss function at once when training our model. Also, it shuffles the data points during each iteration (`shuffle=true`).







## Step 2: Defining the Model
Now that we have our data preprocessed, we can define our model. We will use the Flux.jl package to define our LeNet-5 model.

I like to create a function that I can resue for other image datasets so here is the function. 

```julia
function LeNet5(; imgsize = (32, 32, 3), nclasses = 10)
    in_channels  = imgsize[end]  # for CIFAR-10 is 3 for MNIST Is 1
    #Conv((K,K), in=>out, acivation ) where K is the kernal size
    return Chain(
        Conv((5, 5), in_channels => 6*in_channels, relu),  #pad=(1, 1), stride=(1, 1)),
        MaxPool((2, 2)),
        Conv((5, 5), 6*in_channels=> 16*in_channels, relu),
        MaxPool((2, 2)),
        flatten,
        # Dense(prod(out_conv_size), 120, relu),
        Dense(16*5*5*in_channels=>  120*in_channels, relu),        
        Dense(120*in_channels=> 84*in_channels, relu),
        Dense(84*in_channels=>  nclasses),
    )
end
```
It is a bit different from PyTorch as here, you have to define your kernal. 
If you test it , you should get something like:

```julia-repl
julia> LeNet5()
Chain(
  Conv((5, 5), 3 => 18, relu),          # 1_368 parameters
  MaxPool((2, 2)),
  Conv((5, 5), 18 => 48, relu),         # 21_648 parameters
  MaxPool((2, 2)),
  Flux.flatten,
  Dense(1200 => 360, relu),             # 432_360 parameters
  Dense(360 => 252, relu),              # 90_972 parameters
  Dense(252 => 10),                     # 2_530 parameters
)                   # Total: 10 arrays, 548_878 parameters, 2.095 MiB.

```

## Step 3: Train the model

This section is closly inspired by https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl


First we need to define a struct to hold all of the arguments for the train. (When I could in Python, I usually pass them at the command line but in Julia, it is easier to do so in struct)

### argument 
```julia
Base.@kwdef mutable struct Args
    η = 3e-4             ## learning rate
    λ = 0                ## L2 regularizer param, implemented as weight decay
    batchsize = 128      ## batch size
    epochs = 10          ## number of epochs
    seed = 0             ## set seed > 0 for reproducibility
    use_cuda = true      ## if true use cuda (if available)
    infotime = 1      ## report every `infotime` epochs
    checktime = 5        ## Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      ## log training with tensorboard
    savepath = "runs/"   ## results path
end
```


For the loss function, there are many choises but, here we choose the simplest one.
### Loss function
We use the function [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) to compute the difference between the predicted and actual values (loss).
```julia
loss(ŷ, y) = logitcrossentropy(ŷ, y)

# To output the loss and the accuracy during training:

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]        
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end
```

### Utility functions
 e need a couple of functions to obtain the total number of the model's parameters. Also, we create a function to round numbers to four digits.
```julia
num_params(model) = sum(length, Flux.params(model)) 
round4(x) = round(x, digits=4)
```

### Train the model

```julia
function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    # here we decide to use GPU or not, CUDA.functional() returns true if GPU is detected
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA is loaded 
    train_loader, test_loader = get_data(args)
    @info "Dataset CIFAR-10: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = LeNet5() |> device
    @info "LeNet5 model: $(num_params(model)) trainable params"    
    
    ps = Flux.params(model)  

    # here we use ADAM optimizer but we can change that to any type of supported optimizers

    opt = ADAM(args.η) 
    if args.λ > 0 ## add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.λ), opt)
    end
    
    ## LOGGING UTILITIES
    if args.tblogger 
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) ## 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end
    
    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end
    
    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                    ŷ = model(x)
                    loss(ŷ, y)
                end

            Flux.Optimise.update!(opt, ps, gs)
        end
        
        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson") 
            let model = cpu(model) ## return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end
```



The function `train` performs the following tasks:

* Checks whether there is a GPU available and uses it for training the model. Otherwise, it uses the CPU.
* Loads the CIFAR-10 data using the function `get_data`.
* Creates the model and uses the [ADAM optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAM) with weight decay.
* Loads the [TensorBoardLogger.jl](https://github.com/JuliaLogging/TensorBoardLogger.jl) for logging data to Tensorboard.
* Creates the function `report` for computing the loss and accuracy during the training loop. It outputs these values to the TensorBoardLogger.
* Runs the training loop using [Flux’s training routine](https://fluxml.ai/Flux.jl/stable/training/training/#Training). For each epoch (step), it executes the following:
  * Computes the model’s predictions.
  * Computes the loss.
  * Updates the model’s parameters.
  * Saves the model `model.bson` every `checktime` epochs (defined as argument above.)

## Run the example 
call train()



## Compare with PyTorch

Now that you have trained and evaluated the model with Flux.jl, you can compare its performance to that of a pytorch model trained on CIFAR-10.

To do this, you can use the same code you used to train the model in Flux.jl, but this time using PyTorch. You can then evaluate the model with the same evaluate() function from Flux.jl.

Once you have evaluated both models, you can compare their performance and see which one performs better.

## Reference

* https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl
* Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, november 1998.