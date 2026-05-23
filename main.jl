using Gym
using Flux
using Distributions
using MLUtils

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
        return rand(action_rng, Gym.action_space(env))
    end

    values = model(phi)
    best_idx = argmax(values)
    (values[best_idx], best_idx - 1)
end

function init_network()
    model = Chain(
        Dense(20 => 4, relu),
        Dense(4 => 4, relu),
        Dense(4 => 2))

    optimizer = Flux.setup(Flux.Adam(0.01), model)

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
    experience[4][5] == 1
end

function calc_pred(
    gamma::Float64,
    experience::Experience,
    model
)
    phi = experience[3]
    phi_next = experience[5]
    if is_terminal(experience)
        return phi
    else
        return phi + gamma * max(model(phi_next))
    end
end

function calc_loss(pred, value)
    abs2(pow((pred - value)))
end


function train(
    capacity::Int,
    experiences::Int64,
    steps::Int64,
    epsilon::Float32,
    buff_size::Int,
    gamma::Float64
)

    history = History{Experience}(buff_size)
    model, optimizer = init_network()

    sample_size = 100

    for experience in 1:experiences
        state = [init_state]
        phi = process(4, state)
        Gym.reset!(env)
        for t in 1:steps
            # perform next environment/action step
            value, action = select_action(epsilon, model, phi)
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
                for (x, y) in data
                    y_pred = max(m(x[3]))
                    Flux.mse(y_pred, y)
                end
            end
            Flux.update(optimizer, model, grads)

            if terminated || truncated
                break
            end
        end



    end

end


train(1, 100, 100, 0.1f0, 100, 0.99)

Gym.close(env)
