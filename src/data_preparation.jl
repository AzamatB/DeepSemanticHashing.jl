using Serialization

function load(file_path::AbstractString)
    file_stream = open(file_path, "r")
    structure = deserialize(file_stream)
    close(file_stream)
    return structure
end
