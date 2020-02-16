
# first, we set up a very general function for how we want a trial of our simulation to work
# we don't need to annotate this with any types

"""
    run_trial(observer, task)

Have an `observer` execute a single trial of some `task`
"""
function run_trial(observer, task)
    decision = decide(observer, task)
    outcome = determine_outcome(task, decision)

    # save decision and outcome in a Trial struct
    trial = Trial(decision, outcome)

    # have the observer learn from the trial
    updated_observer = learn(observer, task, trial)

    # return observer and trial
    (observer = updated_observer, trial = trial)
end






"""
An abstract representation of the outcome of one trial
"""
struct Trial{S, T}
    decision::S # what did the observer decide?
    outcome::T # and what outcome did the decision bring?
end


# let's try creating some random Trial instances
Trial("a", "win")
Trial(1, "100 dollars")
Trial(5.0, false)








# now we define some function stubs that don't do anything
# this not a necessary step but just so that we have more
# descriptive errors in the next step (because Julia knows that
# these functions exist, even if they don't do anything yet / have no
# methods)


"""
    decision = decide(observer, task)

Make a `decision` given an `observer` and a `task`.
"""
function decide end






"""
    outcome = determine_outcome(task, decision)

Determine a `task` `outcome given a task and a decision.
"""
function determine_outcome end






"""
    updated_observer = learn(observer, task, trial)

Create an `updated_observer` from an `observer` that has learned
from a `trial` result in a specific `task`.
"""
function learn end







# now we define the logic of a whole experimental run
# again, we don't have to use any type annotations (almost)

function simulate_experiment(observer, task, ntrials)

    observer, first_trial = run_trial(observer, task)

    # a pre-allocated (empty) vector of the correct trial type
    trials = Vector{typeof(first_trial)}(undef, ntrials)
    trials[1] = first_trial

    # run through the rest of the trials while updating the observer
    for i in 2:ntrials
        observer, trials[i] = run_trial(observer, task)
    end

    trials
end








# we now have all the abstract parts that we need for a simulation
# lets make the first concrete part, an actual task


using StaticArrays


"""
A task with N slots that each have a winning probability between 0 and 1.
"""
struct SlotsTask{N}
    probs::SVector{N, Float64} # a vector of probabilities for winning on each slot machine
end


function SlotsTask(probs...)
    N = length(probs)
    SlotsTask(SVector{N, Float64}(probs...))
end




# create a few instances as a test
SlotsTask(0.0, 1.0)
SlotsTask(0.0, 0.3, 0.7)
SlotsTask("a", "b")

# now create and save one task that we're going to use soon
slotstask = SlotsTask(0.2, 0.8)






# now we create our first concrete observer type
# it's a random observer and it doesn't have any fields

struct RandomObserver end


# let's try to run the experiment with the random observer and our task

simulate_experiment(RandomObserver(), slotstask, 1_000)

# there is no suitable method available for decide
# we can check which methods are defined for a function

methods(decide)

# let's define one for our current combination of observer and task

decide(::RandomObserver, ::SlotsTask{N}) where N = rand(1:N)

# check the methods again...

methods(decide)

# and try to run the experiment

simulate_experiment(RandomObserver(), slotstask, 1_000)

# we need another method

determine_outcome(g::SlotsTask, decision::Int) = rand() <= g.probs[decision]

# try again...

simulate_experiment(RandomObserver(), slotstask, 1_000)

# and one more...

learn(r::RandomObserver, task, trial) = r


# now it should work!

random_outcomes = simulate_experiment(RandomObserver(), slotstask, 1_000)



decision(t::Trial) = t.decision
outcome(t::Trial) = t.outcome

using StatsBase

countmap(decision.(random_outcomes))





# let's create a different kind of observer that actually has a strategy


"""
An observer that switches choices every time he loses.
"""
struct SwitchOnLose{T<:Trial}
    lasttrial::Union{Nothing, T} # the observer has to "remember" only the last trial
end

# let's just save our specific trial type for convenience
const SlotTrial = Trial{Int, Bool}

simulate_experiment(SwitchOnLose{SlotTrial}(nothing), slotstask, 1_000)

methods(decide)

function decide(obs::SwitchOnLose, ::SlotsTask{N}) where N

    if isnothing(obs.lasttrial)
        return rand(1:N)
    end

    if determine_outcome(obs.lasttrial) == true
        decision(obs.lasttrial)
    else
        options = tuple((i for i in 1:N if i != decision(obs.lasttrial))...)
        rand(options)
    end
end

methods(decide)


function learn(obs::SwitchOnLose{T}, ::SlotsTask{N}, trial) where {T, N}
    SwitchOnLose{T}(trial)
end

sol_outcomes = simulate_experiment(SwitchOnLose{SlotTrial}(nothing), slotstask, 1000)

countmap(decision.(sol_outcomes))









# now, let's make a third observer that actually comes closer to one used in "real"
# modelling, a simple Rescorla-Wagner observer



struct RescorlaWagnerObserver{N, T}
    α::Float64 # learning rate
    τ::Float64 # softmax temperature
    expvalues::SVector{N, T} # the values currently assigned to the different choices
end

# test creation
RescorlaWagnerObserver(0.3, 1.0, SVector(0.0, 0.0))






function decide(reswa::RescorlaWagnerObserver{N}, ::SlotsTask{N}) where N

    # calculate the probabilities for each choice using the softmax
    # function with the observer's tau value and expected values
    expterms = exp.(reswa.τ .* reswa.expvalues)
    choice_probabilities = expterms ./ sum(expterms)


    r = rand()

    lowerbound = 0.0
    upperbound = choice_probabilities[1]

    # find the bin that the random number is in, that bin is
    # our choice
    for i in 1:N
        # is the random number in the current bin?
        if lowerbound <= r <= upperbound
            # return the choice once we find the correct bin
            return i
        else
            # next bin
            lowerbound += choice_probabilities[i]
            upperbound += choice_probabilities[i]
        end
    end
    return N # just in case the upper bound is not reached due to floating point imprecision
end


methods(decide)



using Setfield

function learn(reswa::RescorlaWagnerObserver{N}, ::SlotsTask{N}, trial) where N
    i = trial.decision

    # convert the true / false outcomes into the values 1.0 / -1.0
    value = trial.outcome == true ? 1.0 : -1.0
    # calculate the prediction error
    pred_error = value - reswa.expvalues[i]

    # now construct a new vector of expected values
    old_expvalues = reswa.expvalues
    # make a new expected values vector with the chosen value updated
    # using the rescorla wagner rule
    new_expvalues = @set old_expvalues[i] += reswa.α * pred_error

    # return a new observer with the new expected values
    @set reswa.expvalues = new_expvalues
end





reswa_outcomes = simulate_experiment(
        RescorlaWagnerObserver(0.9, 1.0, SVector(0.0, 0.0)),
        slotstask,
        1000)

countmap(decision.(reswa_outcomes))







# now just a short demonstration how fast these generically written methods are


using BenchmarkTools

r = RescorlaWagnerObserver(0.9, 1.0, SVector(0.0, 0.0))
tr = Trial(2, false)


# let's time our learn function

@btime learn($r, $slotstask, $tr)



# it's super fast! we can actually look at the type signatures in the whole function body
# to learn whether Julia could infer all types correctly

@code_warntype learn(r, slotstask, tr)



@btime decide(r, slotstask)



@btime simulate_experiment($r, $slotstask, 1000);


# let's try a rescorla wagner experiment with 10_000_000 trials
simulate_experiment(RescorlaWagnerObserver(0.9, 1.0, SVector(0.0, 0.0)), slotstask, 10_000_000);


# or let's try a grid of 10_000 RescorlaWagnerObservers with different learning rates
# and temperatures, with 1000 trials each?
alphas = LinRange(0, 1, 100)
taus = LinRange(0.1, 20, 100)

results = [simulate_experiment(RescorlaWagnerObserver(α, τ, SVector(0.0, 0.0)), slotstask, 1000)
    for α in alphas, τ in taus];

typeof(results)
size(results)

results[1, 1]


using Plots

heatmap(
    alphas,
    taus,
    mean.(outcome.(r) for r in results),
    xlabel = "alpha",
    ylabel = "tau",
    title = "rescorla wagner success rate")









# and now just a quick demonstration of something more R-like with DataFrames
# and plotting them


using DataFrames
using FilteredGroupbyMacro
using Statistics


# first we make a big data frame with all combinations of three tasks and four
# observers, with 100 runs for each

df = join(
    # three tasks
    DataFrame(
        task = [
            SlotsTask(0.1, 0.9),
            SlotsTask(0.25, 0.75),
            SlotsTask(0.4, 0.6)]),
    # four observers
    DataFrame(
        observer = [
            RandomObserver(),
            SwitchOnLose{SlotTrial}(nothing),
            RescorlaWagnerObserver(0.9, 1.0, SVector(0.0, 0.0)),
            RescorlaWagnerObserver(0.05, 1.0, SVector(0.0, 0.0)),
        ]),
    # 100 runs
    DataFrame(
        run = 1:100
    ),
    # all combinations of the above
    kind = :cross
)


# now we run an experiment with 1000 trials for every combination

outcomedf = @by df[!, [:task, :observer, :run],
    trial = simulate_experiment(:observer[1], :task[1], 1000)]

# and store outcomes and decisions in separate columns

outcomedf[:outcome] = outcome.(outcomedf[:trial])
outcomedf[:decision] = decision.(outcomedf[:trial])


# let's find out how many wins the participants could accumulate across
# their 100 runs

avg_outcomes = @by outcomedf[
        !, [:task, :observer, :run],
        p_win = mean(:outcome)
    ][
        !, [:task, :observer],
        mean_p_win = mean(:p_win), sd_p_win = std(:p_win)
    ]


using StatsPlots

@df avg_outcomes groupedbar(
    string.(:task),
    :mean_p_win,
    yerr = :sd_p_win,
    group = string.(:observer),
    ylabel = "Percentage of successes",
    xrotation = 30,
    ylims = (0, 1),
    size = (1000, 700),
    legend = :topright,
    title = "Tasks and strategies")

