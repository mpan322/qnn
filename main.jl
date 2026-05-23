using Gym
using Flux
using Distributions

include("history.jl")

env = Gym.make("CartPole-v1"; render_mode=:human)
action_rng = Gym.seed(123)
init_state, info = Gym.reset!(env)
init_state = push!(init_state, 1)

function select_action(
    epsilon,
    model,
    phi
)
    if rand() < epsilon
        space =  Gym.action_space(env)
        return rand(action_rng, Gym.action_space(env))
    end

    values = model(phi)
    _, index = findmax(values)
    index - 1
end

function init_network()
    model = Chain(
        Dense(20 => 8, relu),
        Dense(8 => 4, relu),
        Dense(4 => 2))

    optimizer = Flux.setup(Flux.Adam(0.001), model)

    model, optimizer
end

# garbage ass inefficient code :)
function process(n, state)
    final = last(state, n)
    init = reshape(stack(final), :)
    n = size(init, 1)
    if n < 20
        diff = 20 - n
        return reshape([init; zeros(diff, 1)], :)
    end
    return reshape(init, :)
end

Experience = Tuple{Vector{Float64},Int64,Float64,Vector{Float64}}

function is_terminal(experience::Experience)
    experience[4][end] == 1
end

function calc_pred(
    gamma::Float64,
    experience::Experience,
    model
)
    phi = experience[3]
    phi_next = experience[4]
    if is_terminal(experience)
        return phi
    else
        return phi + gamma * maximum(model(phi_next))
    end
end

function calc_loss(pred, value)
    abs2(pow((pred - value)))
end


function train(
    experiences::Int64,
    steps::Int64,
    epsilon::Float32,
    sample_size::Int,
    buff_size::Int,
    gamma::Float64
)

    history = History{Experience}(buff_size)
    model, optimizer = init_network()

    for experience in 1:experiences
        state = [init_state]
        phi = process(4, state)
        Gym.reset!(env)
        for t in 1:steps
            # perform next environment/action step
            action = select_action(epsilon, model, phi)
            observation, reward, terminated, truncated, _ = Gym.step!(env, action)
            push!(observation, Int64(terminated || truncated))

            # calculate the experience for this transition
            state = push!(state, observation)
            phi_next = process(4, state)
            experience = (phi, action, reward, phi_next)
            phi = phi_next
            Add!(history, experience)

            # collect minibatch data from history sample
            data = []
            samples = Sample!(sample_size, history)
            for sample in samples
                y = calc_pred(gamma, sample, model)
                push!(data, (sample, y))
            end

            # perform one gradient step
            _, grads = Flux.withgradient(model) do m
                acc = 0
                for (x, y) in data
                    phi_values = m(x[1])
                    y_pred = phi_values[x[2] + 1]
                    acc += Flux.mse(y_pred, y)
                end
                acc
            end
            Flux.update!(optimizer, model, grads[1])

            if terminated || truncated
                break
            end
        end



    end

end


train(600, 400, 0.2f0, 200, 2000, 0.99)

Gym.close(env)
