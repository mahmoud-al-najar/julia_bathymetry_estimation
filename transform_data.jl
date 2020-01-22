using NPZ
using CSV
using DataFrames
using Random
using FileIO
using HDF5


# Parameter setup
csv_path = "C:/Users/Al-Najar/PycharmProjects/bathymetry_estimation/models/v8/data/v8_raw_40x41.csv"
dataset_size = 10
random_seed = 1
Random.seed!(random_seed)

# Dataset preparation
dataset_csv = CSV.File(csv_path; header=["col_path", "col_depth", "col_hmo", "col_freqpeak", "col_thetapeak"])
df = DataFrame(dataset_csv)
# df = df[shuffle(1:size(df, 1)),:]  # Shuffling
# df = df[1:dataset_size, [:col_path, :col_depth]]  # Reducing

for i in 1:size(df, 1)
    path = df.col_path[i]
    println(string(i, " -- ", size(df)))
    npyfile = npzread(string(path, ".npy"))
    h5open(string(path, ".h5"), "w") do file
        write(file, "data", npyfile)  # alternatively, say "@write file A"
    end
end
