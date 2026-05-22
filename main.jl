using Gym
using Flux
using Distributions

env = Gym.make("CartPole-v1"; render_mode=:human)
action_rng = Gym.seed(123)
init_state, info = Gym.reset!(env)

State = Array{Union{AbstractMatrix{Float32}, Bool}, 4}

# struct Episode where
#     curr::State,
#     action::Bool,
#     reward::Float32,
#     next::State
# end

function select_action(
        epsilon,
        model,
        state
    )
    if rand() < epsilon
        return rand(action_rng, Gym.action_space(env))
    end
    Int(round(only(model(only(state)))))
end

function init_network()
    model = Chain(
    Dense(4 => 4, relu),
    Dense(4 => 4, relu),
    Dense(4 => 1, sigmoid))

    optimizer = Flux.setup(Flux.Adam(0.01), model)

    model, optimizer
end


function train(
        capacity::Int,
        episodes::Int64,
        steps::Int64,
        epsilon::Float32,
    )

    # init_memory(capacity)
    model, optimizer = init_network()

    sample_size = 100

    for episode in 1:episodes 
        state = [init_state]
        # phi = process(state)
        Gym.reset!(env)
        for t in 1:steps 

            action = select_action(epsilon, model, state)
            observation, reward, terminated, truncated, info = Gym.step!(env, action)

            if terminated || truncated
                break
            end
            # exec action

            # update state / phi

            # store in circular buffer

            # sample circular buffer

            # y_j = ...

            # gradient = calc_gradient()
            # update_wi
        end

        

    end

end


train(1, 100, 100, 0.1f0)

Gym.close(env)
