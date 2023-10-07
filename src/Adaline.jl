using Random

mutable struct Adaline
  amostras::Vector{Vector{Float64}}
  saidas::Vector{Int}
  taxa_aprendizado::Float64
  taxa_precisao::Float64
  epocas::Int
  limiar::Int
  n_amostras::Int
  n_atributos::Int
  pesos::Vector{Float64}
end
  
function Adaline(amostras, saidas, taxa_aprendizado=0.1, taxa_precisao=0.1, epocas=100, limiar=-1)
  n_amostras = length(amostras)
  n_atributos = length(amostras[1])
  pesos = []
  push!(pesos, limiar)
  append!(pesos, [rand() for _ in 1:(n_atributos + 1)])

  return Adaline(amostras, saidas, taxa_aprendizado, taxa_precisao, epocas, limiar, n_amostras, n_atributos, pesos)
end

function train(rede::Adaline)
  for amostra in rede.amostras
    insert!(amostra, 1, -1)
  end

  n_epocas = 0
  mse_c = mse(rede)
  mse_p = mse(rede)

  while n_epocas <= rede.epocas
    # mse_p = mse_c
    potencial_ativacao = []

    for i in 1:rede.n_amostras
      u = 0.0
      for j in 1:(rede.n_atributos + 1)
        u += rede.pesos[j] * rede.amostras[i][j]
      end
      push!(potencial_ativacao, u)

      for k in 1:(rede.n_atributos + 1)
        rede.pesos[k] += rede.taxa_aprendizado * (rede.saidas[i] - u) * rede.amostras[i][k]
      end
    end

    n_epocas += 1

    if abs(mse_c - mse_p) <= rede.taxa_precisao
      break
    end
  end

  println("Epoch $n_epocas")
  println(abs(mse_c - mse_p))
end

function test(rede::Adaline, amostra::Vector{Float64})
  insert!(amostra, 1, -1)
  u = 0.0
  for i in 1:(rede.n_atributos + 1)
    u += rede.pesos[i] * amostra[i]
  end
  y = signal(u)
  println("Classe: $y")
end

function mse(rede::Adaline)
  eqm = 0.0
  for i in 1:rede.n_amostras
    u = 0.0
    for j in 1:(rede.n_atributos + 1)
      u += rede.pesos[j] * rede.amostras[i][j]
    end
    eqm += (rede.saidas[i] - u)^2
  end
  eqm /= rede.n_amostras
  return eqm
end

function signal(u)
  if u > 0
    return 1
  elseif u == 0
    return 0
  else
    return -1
  end
end
