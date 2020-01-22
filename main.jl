using Flux, Flux.Data.MNIST
using NPZ
using CSV
using DataFrames
using Random
using MLDataUtils
using FileIO, MappedArrays


# Parameter setup
csv_path = "C:/Users/Al-Najar/PycharmProjects/bathymetry_estimation/models/v8/data/v8_raw_40x41.csv"
dataset_size = 10
random_seed = 1
Random.seed!(random_seed)

# Dataset preparation
dataset_csv = CSV.File(csv_path; header=["col_path", "col_depth", "col_hmo", "col_freqpeak", "col_thetapeak"])
df = DataFrame(dataset_csv)
df = df[shuffle(1:size(df, 1)),:]  # Shuffling
df = df[1:dataset_size, [:col_path, :col_depth]]  # Reducing

for i in 1:size(df, 1)
    df.col_path[i] = string(df.col_path[i], ".h5")
end

# Make data generator
@info("Testing data generator")
function load_dataset(dataset)
    x = dataset.col_path
    y = dataset.col_depth / 10
    println(x)
    return mappedarray(load, x), y
end
x_data, y_data = load_dataset(df);
(x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data), at = 0.8)
println(size(x_train))
exit()
# Model construction
model = Chain(
    Conv((3, 3), 4=>16, pad=(1, 1), relu),
    Conv((3, 3), 16=>32, pad=(1, 1), relu),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 256),
    Dense(256, 256),
    Dense(256, 1),
    relu
)

opt = ADAM(1e-05, (0.99, 0.999))
loss(x, y) = Flux.mse(m(x), y)
ps = Flux.params(m)


# Flux.train!(loss, ps, data, opt)
