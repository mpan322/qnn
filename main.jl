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
        return rand(action_rng, Gym.action_space(env))
    end
    Int(round(only(model(phi))))
end

function init_network()
    model = Chain(
        Dense(20 => 4, relu),
        Dense(4 => 4, relu),
        Dense(4 => 1, sigmoid))

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

function train(
    capacity::Int,
    experiences::Int64,
    steps::Int64,
    epsilon::Float32,
    buff_size::Int
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
            action = select_action(epsilon, model, phi)
            observation, reward, terminated, truncated, _ = Gym.step!(env, action)
            push!(observation, Int64(terminated || truncated))

            # calculate the experience for this transition
            state = push!(state, observation)
            phi_next = process(4, state)
            experience = (phi, action, reward, phi_next)
            phi = phi_next
            Add!(history, experience)

            # minibatch GD on sample from history
            samples = Sample!(sample_size, history)

            for sample in samples
                println("SAMPLE!")
            end
            # y_j = ...

            # gradient = calc_gradient()
            # update_wi

            if terminated || truncated
                break
            end
        end



    end

end


train(1, 100, 100, 0.1f0, 100)

Gym.close(env)
