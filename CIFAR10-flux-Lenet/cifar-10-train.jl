using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import MLDatasets
import BSON
using CUDA

# insired by https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl

# ## Data

function get_data(batchsize; dataset = MLDatasets.CIFAR10, idxs = nothing, T = Float32)
    """
    idxs=nothing gives the full dataset, otherwise (for testing purposes) only the 1:idxs elements of the train set are given.
    dataset is the datasets we will use

    """
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset
    if idxs === nothing
        xtrain, ytrain = dataset(Tx = T, :train)[:]
        xtest, ytest = dataset(Tx = T, :test)[:]
    else
        xtrain, ytrain = dataset(Tx = T, :train)[1:idxs]
        xtest, ytest = dataset(Tx = T, :test)[1:Int(idxs / 10)]
    end

    # Reshape Data to comply to Julia's (width, height, channels, batch_size) convention in case there are only 1 channel (eg MNIST)
    if ndims(xtrain) == 3
        w = size(xtrain)[1]
        xtrain = reshape(xtrain, (w, w, 1, :))
        xtest = reshape(xtest, (w, w, 1, :))
    end

    # construct one-hot vectors from labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true)
    test_loader = DataLoader((xtest, ytest), batchsize = batchsize)

    return train_loader, test_loader
end



function LeNet5(; imgsize = (32, 32, 3), nclasses = 10)
    # out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)
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








# We set default values for the arguments for the function `train`:

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

# ## Loss function

# We use the function [logitcrossentropy](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitcrossentropy) to compute the difference between 
# the predicted and actual values (loss).

loss(ŷ, y) = logitcrossentropy(ŷ, y)

# Also, we create the function `eval_loss_accuracy` to output the loss and the accuracy during training:

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

# ## Utility functions
# We need a couple of functions to obtain the total number of the model's parameters. Also, we create a function to round numbers to four digits.

num_params(model) = sum(length, Flux.params(model)) 
round4(x) = round(x, digits=4)

# ## Train the model


# Finally, we define the function `train` that calls the functions defined above to train the model.

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    train_loader, test_loader = get_data(args)
    @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = LeNet5() |> device
    @info "LeNet5 model: $(num_params(model)) trainable params"    
    
    ps = Flux.params(model)  

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


# ## Run the example 
# call train()