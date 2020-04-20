using Flux
using CSV
using DataFrames
using Random
using MLDataUtils
using Statistics
using Printf
using BSON
include("dataloader.jl")

using CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
else
    @error "Couldn't find CUDA"
    exit()
end

# Parameter setup
csv_path = "" # ------------------------ TO BE SET
dataset_size = 10_000
batch_size = 64
epochs = 50
patience = 10
train_test_split = 0.8
random_seed = 2
Random.seed!(random_seed)

@info("Preparing the dataset")
dataset_csv = CSV.File(csv_path; header=["col_path", "col_depth", "col_hmo", "col_freqpeak", "col_thetapeak"])
df = DataFrame(dataset_csv)
df = df[shuffle(1:size(df, 1)),:]  # Shuffling
df = df[1:dataset_size, [:col_path, :col_depth]]  # Reducing

for i in 1:size(df, 1)
    # CSV doesn't have the file extension
    df.col_path[i] = string(df.col_path[i], ".h5")
end

# Make data generator
function load_dataset(dataset)
    x = dataset.col_path
    y = dataset.col_depth / 10
    return x,  convert(Array{Float32}, y)  # Y
end

x_data, y_data = load_dataset(df);
(x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data), at = train_test_split)
trainloader = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
testloader = DataLoader((x_test, y_test), batchsize=batch_size, shuffle=true)

@info("Building the model")
model = Chain(
    Conv((3, 3), 4=>16, pad=(1, 1), relu),
    Conv((3, 3), 16=>32, pad=(1, 1), relu),
    x -> reshape(x, :, size(x, 4)),
    Dense(51200, 256, relu),
    Dense(256, 256, relu),
    Dense(256, 1, relu)
)

# Load onto GPU if available
# @info("Loading onto GPU")
# model = gpu(model)
# trainloader = gpu(trainloader)
# testloader = gpu(testloader)

opt = ADAM(1e-05, (0.99, 0.999))

function loss(x, y)
    return Flux.mse(model(x), y)
end

function evaluate(testloader)
    errors = []
    for (x, y) in testloader
        mse = Flux.mse(cpu(model(x)), cpu(y))
        append!(errors, mse)
    end
    return mean(errors)
end

@info("Training")
best_val = 999
last_improvement = 0
for epoch_idx in 1:epochs
    global best_val, last_improvement
    Flux.train!(loss, Flux.params(model), trainloader, opt)

    new_val = evaluate(testloader)
    @info(@sprintf("[%d]: Validation MSE: %.4f", epoch_idx, new_val))
    if new_val < best_val
        best_val = new_val
        last_improvement = epoch_idx
        filename = string(epoch_idx, "__", new_val, ".bson")
        @info(" -> New best! Saving model out to ", filename)
        BSON.@save joinpath(dirname(@__FILE__), filename) model epoch_idx new_val
    end

    # Learning rate scheduler
    if epoch_idx == 40 || epoch_idx == 60 || epoch_idx == 80
        opt.eta /= 10.0
        @info(@sprintf("Epoch [%d]: Dropping learning rate to %.1e", epoch_idx, opt.eta))
    end

    # Early stopping
    if epoch_idx - last_improvement > patience
        Flux.stop()
    end

end
