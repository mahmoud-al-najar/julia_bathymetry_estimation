using Random: randperm!
using HDF5

"""
    MODIFIED DataLoader Class
    Modifications:
      - Altered "Base.iterate" implementation to handle hdf5 (.h5) files
      - Files are only read from disk when they're needed

    Original version: https://github.com/boathit/Benchmark-Flux-PyTorch/blob/master/dataloader.jl
    Original documentation:
    DataLoader(dataset::AbstractArray...; batchsize::Int, shuffle::Bool)
DataLoader provides iterators over the dataset.
```julia
X = rand(10, 1000)
Y = rand(1, 1000)
m = Dense(10, 1)
loss(x, y) = Flux.mse(m(x), y)
opt = ADAM(params(m))
trainloader = DataLoader(X, Y, batchsize=256, shuffle=true)
Flux.train!(loss, trainloader, opt)
```
"""
struct DataLoader
  dataset::Tuple
  batchsize::Int
  shuffle::Bool
  indices::Vector{Int}
  n::Int
end

function DataLoader(
  dataset::Tuple{AbstractArray,Vararg{AbstractArray}};
  batchsize::Int,
  shuffle::Bool,
)
  l = last.(size.(dataset))
  n = first(l)
  all(n .== l) || throw(DimensionMismatch("All data should have the same length."))
  indices = collect(1:n)
  shuffle && randperm!(indices)
  DataLoader(dataset, batchsize, shuffle, indices, n)
end

DataLoader(dataset::AbstractArray...; batchsize::Int, shuffle::Bool) =
  DataLoader(dataset, batchsize = batchsize, shuffle = shuffle)

function Base.iterate(it::DataLoader, start = 1)
  if start > it.n
    it.shuffle && randperm!(it.indices)
    return nothing
  end
  nextstart = min(start + it.batchsize, it.n + 1)
  i = it.indices[start:nextstart-1]

  # Select batch data
  raw_batch = Tuple(copy(selectdim(x, ndims(x), i)) for x in it.dataset)

  # Prepare empty batch arrays of size (dim1, dim2 .... dimN, 1)
  # Added dimension: index in batch
  X_batch = Array{Float32}(undef, 40, 40, 4, it.batchsize)
  Y_batch = Array{Float32}(undef, 1, it.batchsize)

  for i in 1:it.batchsize
    # raw_batch[1][i] here includes the path to the h5 file to be read
    img = permutedims(h5open(raw_batch[1][i], "r")["data"][1:4,:,1:40], [2, 3, 1])
    X_batch[:, :, :, i] = img

    depth = raw_batch[2][i]
    Y_batch[1, i] = depth
  end

  new_element = (X_batch, Y_batch)

  return new_element, nextstart
end

Base.length(it::DataLoader) = it.n
Base.eltype(it::DataLoader) = typeof(it.dataset)

function Base.show(io::IO, it::DataLoader)
  print(io, "DataLoader(dataset size = $(it.n)")
  print(io, ", batchsize = $(it.batchsize), shuffle = $(it.shuffle)")
  print(io, ")")
end
