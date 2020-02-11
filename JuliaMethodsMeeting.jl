struct Trial{S, T}
    decision::S
    result::T
end


function run_trial(strategy, game, previous_trials)
    decision = decide(strategy, game, previous_trials)
    result = determine_result(game, decision)
    Trial(decision, result)
end

function run_experiment(strategy, game, ntrials)

    first_trial = run_trial(strategy, game, [])

    trials = Vector{typeof(first_trial)}(undef, ntrials)
    trials[1] = first_trial

    for i in 2:ntrials
        trials[i] = run_trial(strategy, game, view(trials, 1:i-1))
    end

    trials
end


struct SlotsGame{N}
    probs::NTuple{N, Float64}
end

struct RandomStrategy end


decide(::RandomStrategy, ::SlotsGame{N}, previous_trials) where N = rand(1:N)

determine_result(g::SlotsGame, decision::Int) = rand() > g.probs[decision]

random_results = run_experiment(RandomStrategy(), SlotsGame((0.2, 0.8)), 1_000)

using StatsBase

decision(t::Trial) = t.decision
result(t::Trial) = t.result

countmap(decision.(random_results))


struct SwitchOnLose end

function decide(::SwitchOnLose, ::SlotsGame{N}, previous_trials) where N

    if isempty(previous_trials)
        return rand(1:N)
    end

    p = previous_trials[end]
    if result(p) == true
        decision(p)
    else
        options = tuple((i for i in 1:N if i != decision(p))...)
        rand(options)
    end
end

sol_results = run_experiment(SwitchOnLose(), SlotsGame((0.2, 0.8)), 1000)

countmap(decision.(sol_results))

struct ChooseBestSoFar end

function decide(::ChooseBestSoFar, ::SlotsGame{N}, previous_trials) where N

    choicecounts = zeros(Int, N)
    successes = zeros(Int, N)

    for t in previous_trials
        choicecounts[t.decision] += 1
        if t.result == true
            successes[t.decision] += 1
        end
    end

    successratios = successes ./ choicecounts
    best = argmax(successratios)
end

bsf_results = run_experiment(ChooseBestSoFar(), SlotsGame((0.2, 0.8)), 1000);

countmap(decision.(bsf_results))


results = [run_experiment(st, SlotsGame(probs), 1000)
    for st in (ChooseBestSoFar(), RandomStrategy(), SwitchOnLose()),
        probs in [(0.2, 0.8), (0.5, 0.5), (1/3, 1/3, 1/3)]
];


countmap.(decision.(r) for r in results)

using Distributions

struct DistributionSlots{N, D}
    distributions::NTuple{N, D}
end



dslots = DistributionSlots((Normal(0, 1), Normal(1, 4)))


dist_results = run_experiment(SwitchOnLose(), dslots, 1000)

function decide(::SwitchOnLose, dslots::DistributionSlots{N}, previous_trials) where N
    if isempty(previous_trials)
        return rand(1:N)
    end

    p = previous_trials[end]
    if result(p) > 0
        decision(p)
    else
        options = tuple((i for i in 1:N if i != decision(p))...)
        rand(options)
    end
end

function determine_result(dslots::DistributionSlots{N}, choice) where N
    rand(dslots.distributions[choice])
end

# using DataFrames

# df = DataFrame(decision = decision.(dist_results), result = result.(dist_results))

# by(df, :decision, :result => sum)

results = [run_experiment(SwitchOnLose(), dslots, 100) for i in 1:1000]


using Plots


Plots.plot()

histogram(getindex.(countmap.(decision.(r) for r in results), 1))
