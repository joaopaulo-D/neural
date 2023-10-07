using Random

mutable struct Perceptron
  input::Vector{Vector{Float64}}
  output::Vector{Int64}
  lr::Float64
  interation::Int
  limiar::Int
  n_amostras::Int
  n_atributos::Int
  pesos::Vector{Float64}
end

function Perceptron(input, output, lr, interation, limiar, n_amostras, n_atributos, pesos)
  n_amostras = length(input)
  n_atributos = length(input[1])
  pesos = []
  push!(pesos, limiar)
  append!(pesos, [rand() for _ in 1:(n_atributos + 1)])

  return Perceptron(input, output, lr, interation, limiar, n_amostras, n_atributos, pesos)
end

function train(p::Perceptron)
  for out in p.output
    insert!(out, 0)
  end
end