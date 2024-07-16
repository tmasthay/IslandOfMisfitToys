"""
This script was used to generate the pytorch .pt files for sigsbee given an input segy file.
We plan to make the generation of those pytorch files from the segy files a part of this package directly
    without the need of any external dependencies, but for now, this docstring describes how to generate
    the .pt files with this script.

1. Install Madagascar
    a. Instructions -> https://www.reproducibility.org/wiki/Installation
    b. Source code -> https://github.com/ahay/src
2. Run Sigsbee2A `SConstruct` file
    a. See https://github.com/ahay/src/blob/master/book/data/sigsbee/model2A/SConstruct
    b. As of time of writing, the command `Fetch` does not work in Madagascar due to data hosting issues.
        If `Fetch` does not work, refer instead to the link below instead.
        https://zapad.stanford.edu/taylor/sigsbee-model/-/tree/master
3. Run `get_torch_tensors.jl`
    a. Install julia -> https://julialang.org/downloads/
    b. Run julia -e 'using Pkg; Pkg.add("Madagascar"); Pkg.add("PyCall")'
    c. Run julia get_torch_tensors.jl
"""
using Madagascar
using PyCall

function write_rsf_to_pt(filename::String; save_to_disk::Bool=true)
    torch = pyimport("torch")

    rsf_obj = rsf_read("$filename.rsf")
    data = sfwindow(rsf_obj[1]; f1=1) |> rsf_read |> x -> torch.tensor(x[1]).float()

    save_to_disk && torch.save(data, "$filename.pt")

    return data
end

function reconvert_to_rsf(filename::String)
    torch = pyimport("torch")

    data = torch.load("$filename.pt")
    numpy_data = data.numpy()
    julia_matrix = convert(Matrix{Float32}, numpy_data)
    rsf_write("$(filename)_reconstructed.rsf", julia_matrix)
end

if abspath(PROGRAM_FILE) == @__FILE__
    torch = pyimport("torch")
    plt = pyimport("matplotlib.pyplot")

    filenames = ["vstr2A", "vmig2A"]
    for filename in filenames
        data = write_rsf_to_pt(filename)
        plt.imshow(data, aspect="auto")
        plt.colorbar()
        plt.savefig("$(filename).jpg")
        plt.clf()
    end
end
