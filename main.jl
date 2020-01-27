using Flux, Flux.Data.MNIST
using CSV
using DataFrames
using Random
using MLDataUtils
using FileIO
using HDF5
using Base.Iterators: partition
using Statistics
include("dataloader.jl")

# Parameter setup
csv_path = "C:/Users/Al-Najar/PycharmProjects/bathymetry_estimation/models/v8/data/v8_raw_40x41.csv"
dataset_size = 10
batch_size = 2
epochs = 2
train_test_split = 0.8
random_seed = 2
Random.seed!(random_seed)

@info("Preparing the dataset")
dataset_csv = CSV.File(csv_path; header=["col_path", "col_depth", "col_hmo", "col_freqpeak", "col_thetapeak"])
df = DataFrame(dataset_csv)
df = df[shuffle(1:size(df, 1)),:]  # Shuffling
df = df[1:dataset_size, [:col_path, :col_depth]]  # Reducing

for i in 1:size(df, 1)
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
@info("Loading onto GPU")
model = gpu(model)
trainloader = gpu(trainloader)

opt = ADAM(1e-05, (0.99, 0.999))

function loss(x, y)
    # return mean((model(x) - y) .^ 2)
    return Flux.mse(model(x), y)
end

Flux.train!(loss, Flux.params(model), trainloader, opt)

@info("Training")
best_acc = 999
last_improvement = 0
for epoch_idx in 1:epochs
    global best_acc, last_improvement
    # Train for a single epoch
    println("EPOCH: ", epoch_idx)
    Flux.train!(loss, Flux.params(model), trainloader, opt)
end
