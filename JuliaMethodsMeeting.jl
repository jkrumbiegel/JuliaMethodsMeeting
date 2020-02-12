"""
An abstract representation of the result of one trial
"""
struct Trial{S, T}
    decision::S # what did the observer decide?
    result::T # and what result did the decision bring?
end


function run_trial(strategy, game, previous_trials)
    decision = decide(strategy, game, previous_trials)
    result = determine_result(game, decision)
    # learn here
    # learn!(strategy, game, decision, result)
    Trial(decision, result)
end

function decide(strategy, game, previous_trials)
    error("""
        No decide method yet for strategy $(typeof(strategy)) and game $(typeof(game)).
        You have to create one!)
    """)
end

function determine_result(game, decision)
    error("""
        No determine_result method yet for game $(typeof(game)) and decision $(typeof(decision)).
        You have to create one!)
    """)
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








"""
A game with N slots that each have a winning probability between 0 and 1.
"""
struct SlotsGame{N}
    probs::NTuple{N, Float64}
end

SlotsGame((0.0, 1.0))
SlotsGame((0.0, 0.3, 0.7))

slotsgame = SlotsGame((0.2, 0.8))








struct RandomStrategy end


run_experiment(RandomStrategy(), slotsgame, 1_000)

decide(::RandomStrategy, ::SlotsGame{N}, previous_trials) where N = rand(1:N)

run_experiment(RandomStrategy(), slotsgame, 1_000)

determine_result(g::SlotsGame, decision::Int) = rand() <= g.probs[decision]

random_results = run_experiment(RandomStrategy(), slotsgame, 1_000)


decision(t::Trial) = t.decision
result(t::Trial) = t.result

using StatsBase

countmap(decision.(random_results))









struct SwitchOnLose end

run_experiment(SwitchOnLose(), slotsgame, 1_000)


function decide(::SwitchOnLose, ::SlotsGame{N}, previous_trials) where N

    if isempty(previous_trials)
        return rand(1:N)
    end

    lasttrial = previous_trials[end]
    if result(lasttrial) == true
        decision(lasttrial)
    else
        options = tuple((i for i in 1:N if i != decision(lasttrial))...)
        rand(options)
    end
end

sol_results = run_experiment(SwitchOnLose(), slotsgame, 1000)

countmap(decision.(sol_results))









struct ChooseBestInLastN
    n::Int
end

function decide(strategy::ChooseBestInLastN, ::SlotsGame{N}, previous_trials) where N

    choicecounts = zeros(Int, N)
    successes = zeros(Int, N)

    nprevious = length(previous_trials)
    nback = strategy.n
    indices = max(1, nprevious - nback):nprevious
    for i in indices
        t = previous_trials[i]
        choicecounts[t.decision] += 1
        if t.result == true
            successes[t.decision] += 1
        end
    end

    successratios = successes ./ choicecounts
    best = argmax(successratios)
end

bsf_results = run_experiment(ChooseBestInLastN(10), slotsgame, 1_000);

countmap(decision.(bsf_results))










using DataFrames
using FilteredGroupbyMacro
using Statistics

df = join(
    DataFrame(
        game = [SlotsGame((0.2, 0.8)), SlotsGame((0.4, 0.6)), SlotsGame((0.5, 0.5)), SlotsGame((0.9, 0.6, 0.3))]),
    DataFrame(
        strategy = [
            RandomStrategy(), SwitchOnLose(), ChooseBestInLastN(10), ChooseBestInLastN(3)
        ]),
    DataFrame(
        run = 1:100
    ),
    kind = :cross
)

resultdf = @by df[!, [:game, :strategy, :run],
    trial = run_experiment(:strategy[1], :game[1], 1000)]

resultdf[:result] = result.(resultdf[:trial])
resultdf[:decision] = decision.(resultdf[:trial])


avg_results = @by resultdf[!, [:game, :strategy, :run],
        p_win = mean(:result)
    ][!, [:game, :strategy],
        mean_p_win = mean(:p_win), sd_p_win = std(:p_win)
    ]


using Plots
using StatsPlots

@df avg_results groupedbar(
    string.(:game),
    :mean_p_win,
    yerr = :sd_p_win,
    group = string.(:strategy),
    ylabel = "Percentage of successes",
    xrotation = 30,
    ylims = (0, 1),
    legend = :top,
    title = "Games and strategies")







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

    previous = previous_trials[end]
    if result(previous) > 0
        decision(previous)
    else
        options = tuple((i for i in 1:N if i != decision(previous))...)
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
