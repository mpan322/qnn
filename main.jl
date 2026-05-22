using Gym
using Flux
using Distributions

State = Array{Union{AbstractMatrix{Float32}, Bool}, 4}

struct Episode where
    curr::State,
    action::Bool,
    reward::Float32,
    next::State
end

function select_action(
        epsilon,
        model
    )

    if rand() < epsilon
        # choose random action
    end
    # choose optimal action
end

model = Chain(
    Dense(4 => 4, relu),
    Dense(4 => 4, relu),
    Dense(4 => 1, sigmoid))

optimizer = Flux.setup(Flux.Adam(0.01), model)

function train(
        capacity::Int,
        episodes::Int64,
        steps::Int64,
        epsilon::Float32,
        model
    )

    init_memory(capacity)
    init_network()

    sample_size = 100

    for episode in 1:episodes 
        state = [get_screen()]
        phi = process(state)
        for t in 1:steps 

            action = select_action(epsilon)

            # exec action

            # update state / phi

            # store in circular buffer

            # sample circular buffer

            # y_j = ...

            gradient = calc_gradient()
            update_wi
        end

    end

end


env = Gym.make("CartPole-v1"; render_mode=:human)
observation, info = Gym.reset!(env)

while true
    action = rand(Gym.action_space(env))
    observation, reward, terminated, truncated, info = Gym.step!(env, action)
end

Gym.close(env)
