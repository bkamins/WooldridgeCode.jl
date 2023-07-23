include("init_example.jl")

# Problem C1

wage1 = get_dataset("wage1")
describe(wage1)
count(==(1), wage1.female)
count(==(0), wage1.female)
combine(groupby(wage1, :female), nrow)

# Problem C2

bwght = get_dataset("bwght")
nrow(bwght)
count(>(0), bwght.cigs)
mean(bwght.cigs[bwght.cigs .> 0])
mean(skipmissing(bwght.fatheduc))
count(!ismissing, bwght.fatheduc)
describe(bwght; cols=:fatheduc)
mean(1000 * bwght.faminc)
std(1000 * bwght.faminc)

# Problem C3

meap01 = get_dataset("meap01")
extrema(meap01.math4)
count(==(100), meap01.math4)
mean(meap01.math4 .== 100)
count(==(50), meap01.math4)
describe(meap01, :mean; cols=[:math4, :read4])
cor(meap01.math4, meap01.read4)
describe(meap01, :mean, :std; cols=:exppp)
100 * (6000 / 5500 - 1)
100 * (log(6000) - log(5500))

# Problem C4

jtrain2 = get_dataset("jtrain2")
mean(jtrain2.train)
@combine(groupby(jtrain2, :train), :re78_mean = mean(:re78))
@combine(groupby(jtrain2, :train), :unem78_mean = mean(:unem78))

# Problem C5

fertil2 = get_dataset("fertil2")
describe(fertil2; cols=:children)
describe(fertil2; cols=:electric)
mean(skipmissing(fertil2.electric))
@combine(groupby(fertil2, :electric), :children_mean = mean(:children))

# Problem C6

countymurders = get_dataset("countymurders")
cm1996 = @rsubset(countymurders, :year == 1996)
allunique(cm1996, :countyid)
nrow(cm1996)
count(==(0), cm1996.murders)
mean(cm1996.execs .== 0)
describe(cm1996; cols=[:murders, :execs])
cor(cm1996.murders, cm1996.execs)

# Problem C7

alcohol = get_dataset("alcohol")
describe(alcohol; cols=[:abuse, :employ])
@combine(groupby(alcohol, :abuse), :employ_mean = mean(:employ))

# Problem C8

econmath = get_dataset("econmath")
nrow(econmath)
count(==(1), econmath.econhs)
sum(econmath.econhs)
@combine(groupby(econmath, :econhs), :score_mean = mean(:score))

