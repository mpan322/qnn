using Gym

env = Gym.make("CartPole-v1"; render_mode=:human)
observation, info = Gym.reset!(env)

while true
    action = rand(Gym.action_space(env))
    observation, reward, terminated, truncated, info = Gym.step!(env, action)
    if terminated || truncated
        break
    end
end

Gym.close(env)