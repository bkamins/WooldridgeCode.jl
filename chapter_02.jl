include("init_example.jl")

# Example 2.3

ceosal1 = get_dataset("ceosal1")
salary_roe = lm(@formula(salary ~ roe), ceosal1)
nobs(salary_roe)
predict(salary_roe, DataFrame(roe=30))

# Example 2.4

wage1 = get_dataset("wage1")
wage_educ = lm(@formula(wage ~ educ), wage1)
nobs(wage_educ)
predict(wage_educ, DataFrame(educ=8))

# Example 2.5

vote1 = get_dataset("vote1")
voteA_shareA = lm(@formula(voteA ~ shareA), vote1)
nobs(voteA_shareA)
predict(voteA_shareA, DataFrame(shareA=50))

# Example 2.7

mean(wage1.wage)
mean(wage1.educ)
predict(wage_educ, DataFrame(educ=mean(wage1.educ)))

# Example 2.8

r2(salary_roe)

# Example 2.9

r2(voteA_shareA)

# Example 2-4a

ceosal1.salardol = 1000 * ceosal1.salary
saldol_roe = lm(@formula(salardol ~ roe), ceosal1)
r2(saldol_roe)

ceosal1.roedec = ceosal1.roe / 100
salary_roedec = lm(@formula(salary ~ roedec), ceosal1)
r2(salary_roedec)

# Example 2.10

lwage_educ = lm(@formula(lwage ~ educ), wage1)
nobs(lwage_educ)
r2(lwage_educ)

# Example 2.11

lsalary_lsales = lm(@formula(lsalary ~ lsales), ceosal1)
nobs(lsalary_lsales)
r2(lsalary_lsales)

# an alternative way to estimate the same model
lm(@formula(log(salary) ~ log(sales)), ceosal1)

# Example 2.12

meap93 = get_dataset("meap93")
math_lnchprg = lm(@formula(math10 ~ lnchprg), meap93)
nobs(math_lnchprg)
r2(math_lnchprg)

# Example 2.14

jtrain2 = get_dataset("jtrain2")
re_train = lm(@formula(re78 ~ train), jtrain2)
nobs(re_train)
r2(re_train)

# Problem 3

act_gpa_df = DataFrame(act=[2.8, 3.4, 3.0, 3.5, 3.6, 3.0, 2.7, 3.7],
                       gpa=[21, 24, 26, 27, 29, 25, 25, 30])
gpa_act = lm(@formula(gpa ~ act), act_gpa_df)
nobs(gpa_act)
r2(gpa_act)
act_gpa_df.pred = predict(gpa_act)
act_gpa_df.resid = residuals(gpa_act)
mapcols(mean, act_gpa_df)
predict(gpa_act, DataFrame(act=20))

# Problem 4

bwght = get_dataset("bwght")
bwght_cigs = lm(@formula(bwght~cigs), bwght)
nobs(bwght_cigs)
r2(bwght_cigs)
predict(bwght_cigs, DataFrame(cigs=[0, 20]))

# Problem C1

k401k = get_dataset("k401k")
describe(k401k; cols=[:prate, :mrate])
prate_mrate = lm(@formula(prate~mrate), k401k)
nobs(prate_mrate)
r2(prate_mrate)
predict(prate_mrate, DataFrame(mrate=3.5))

# Problem C2

ceosal2 = get_dataset("ceosal2")
describe(ceosal2; cols=[:salary, :ceoten])
count(==(0), ceosal2.ceoten)
mean(==(0), ceosal2.ceoten)
lsalary_ceoten = lm(@formula(lsalary~ceoten), ceosal2)
nobs(lsalary_ceoten)
r2(lsalary_ceoten)

# Problem C3

sleep75 = get_dataset("sleep75")
sleep_totwrk = lm(@formula(sleep~totwrk), sleep75)
nobs(sleep_totwrk)
r2(sleep_totwrk)

# Problem C4

wage2 = get_dataset("wage2")
describe(wage2, :detailed; cols=[:wage, :IQ])
wage_IQ = lm(@formula(wage~IQ), wage2)
nobs(wage_IQ)
r2(wage_IQ)
lwage_IQ = lm(@formula(lwage~IQ), wage2)
nobs(wage_IQ)
r2(wage_IQ)

# Problem C5

rdchem = get_dataset("rdchem")
lrd_lsales = lm(@formula(lrd~lsales), rdchem)
nobs(lrd_lsales)
r2(lrd_lsales)

# Problem C6

math10_lexpend = lm(@formula(math10 ~ lexpend), meap93)
nobs(math10_lexpend)
r2(math10_lexpend)

# Problem C7

charity = get_dataset("charity")
describe(charity)
mean(==(0), charity.gift)
mean(charity.gift .== 0)
gift_mailsyear = lm(@formula(gift ~ mailsyear), charity)
nrobs(gift_mailsyear)
r2(gift_mailsyear)
predict(gift_mailsyear, DataFrame(mailsyear=[0, minimum(charity.mailsyear)]))

# Problem C8

c8 = DataFrame(x = rand(Uniform(0, 10), 500),
               u = rand(Normal(0, 6), 500))
describe(c8, :detailed)
@rtransform!(c8, :y = 1.0 + 2 * :x + :u)
y_x = lm(@formula(y ~ x), c8)
u_hat = residuals(y_x)
sum(u_hat)
sum(u_hat .* c8.x)
sum(c8.u)
sum(c8.u .* c8.x)

# Repeat the process to ansver point (vi)

# Problem C9

countymurders = get_dataset("countymurders")
cm1996 = @rsubset(countymurders, :year == 1996)
allunique(cm1996, :countyid)
count(==(0), cm1996.murders)
count(>=(1), cm1996.execs)
maximum(cm1996.execs)
murders_execs = lm(@formula(murders ~ execs), cm1996)
nobs(murders_execs)
r2(murders_execs)

# Problem C10

catholic = get_dataset("catholic")
nrow(catholic)
describe(catholic, :mean, :std; cols=[:math12, :read12])
math12_read12 = lm(@formula(math12 ~ read12), catholic)
nobs(math12_read12)
r2(math12_read12)

# Problem C11

gpa1 = get_dataset("gpa1")
nrow(gpa1)
describe(gpa1)
sum(gpa1.PC)
colGPA_PC = lm(@formula(colGPA ~ PC), gpa1)
nobs(colGPA_PC)
r2(colGPA_PC)

