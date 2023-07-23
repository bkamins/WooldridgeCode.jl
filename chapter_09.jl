include("init_example.jl")

# Example 9.1

crime1 = get_dataset("crime1")
narr86_multi1 = lm(@formula(narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + inc86 + black + hispan), crime1)
narr86_multi2 = lm(@formula(narr86 ~ pcnv + pcnv^2 + avgsen + tottime + ptime86 + ptime86^2 + qemp86 + inc86 + inc86^2 + black + hispan), crime1)
ftest(narr86_multi1.model, narr86_multi2.model)

# Example 9.2

hprice1 = get_dataset("hprice1")
price_multi = lm(@formula(price ~ lotsize + sqrft + bdrms), hprice1)
lprice_multi = lm(@formula(lprice ~ llotsize + lsqrft + bdrms), hprice1)
# needed for numerical stability
hprice1.kp = predict(price_multi) ./ 10
hprice1.lp = predict(lprice_multi)

price_multi2 = lm(@formula(price ~ lotsize + sqrft + bdrms + kp^2 + kp^3), hprice1)
lprice_multi2 = lm(@formula(lprice ~ llotsize + lsqrft + bdrms + lp^2 + lp^3), hprice1)

ftest(price_multi.model, price_multi2.model)
ftest(lprice_multi.model, lprice_multi2.model)

# Example 9.3 and Problem C2

wage2 = get_dataset("wage2")

lwage_multi1 = lm(@formula(lwage ~ educ + exper + tenure + married + south + urban + black), wage2)
lwage_multi2 = lm(@formula(lwage ~ educ + IQ + exper + tenure + married + south + urban + black), wage2)
lwage_multi3 = lm(@formula(lwage ~ educ*IQ + exper + tenure + married + south + urban + black), wage2)

lwage_multi4 = lm(@formula(lwage ~ educ + KWW + exper + tenure + married + south + urban + black), wage2)
lwage_multi5 = lm(@formula(lwage ~ educ*KWW + exper + tenure + married + south + urban + black), wage2)
lwage_multi6 = lm(@formula(lwage ~ educ + IQ + KWW + exper + tenure + married + south + urban + black), wage2)
ftest(lwage_multi1.model, lwage_multi6.model)

# Example 9.4

crime2 = get_dataset("crime2")
crime2_87 = @chain crime2 begin
    @rsubset(:year == 87)
    select(:crmrte, :unem, :lawexpc)
end
crime2_87.crmrte82 = crime2.crmrte[crime2.year .== 82]
lm(@formula(log(crmrte) ~ unem + log(lawexpc)), crime2_87)
lm(@formula(log(crmrte) ~ unem + log(lawexpc) + log(crmrte82)), crime2_87)

# Example 9.8 and 9.9 and Problem C5

rdchem = get_dataset("rdchem")
lm(@formula(rdintens ~ sales + profmarg), rdchem)
lm(@formula(rdintens ~ sales + profmarg), rdchem[Not(10), :])
lm(@formula(rdintens ~ sales + profmarg), rdchem[Not(1, 10), :])

lm(@formula(lrd ~ lsales + profmarg), rdchem)
lm(@formula(lrd ~ lsales + profmarg), rdchem[Not(10), :])

# division by 1000 due to numerical stability issues
# TODO: in GLM.jl 2.0 it will not be needed; use method=:qr
rdintens_model1 = lm(@formula(rdintens ~ (sales/1000) + (sales/1000)^2 + profmarg), rdchem)
rdintens_model2 = lm(@formula(rdintens ~ (sales/1000) + (sales/1000)^2 + profmarg), rdchem[Not(10), :])

lad(rdintens_model1)
lad(rdintens_model2)

# Example 9.10 and Problem C4

infmrt = get_dataset("infmrt")
infmrt1990 = @rsubset(infmrt, :year == 1990)

lm(@formula(infmort ~ lpcinc + lphysic + lpopul), infmrt1990)
lm(@formula(infmort ~ lpcinc + lphysic + lpopul), infmrt1990[Not(24), :])

infmrt1990.dc = [i == 24 for i in axes(infmrt1990, 1)]
lm(@formula(infmort ~ lpcinc + lphysic + lpopul + dc), infmrt1990)

# Problem 1

ceosal2 = get_dataset("ceosal2")

lsalary_multi1 = lm(@formula(lsalary ~ lsales + lmktval + profmarg + ceoten + comten), ceosal2)
lsalary_multi2 = lm(@formula(lsalary ~ lsales + lmktval + profmarg + ceoten + ceoten^2 + comten + comten^2), ceosal2)
ftest(lsalary_multi1.model, lsalary_multi2.model)

# Problem 2

vote2 = get_dataset("vote2")
lm(@formula(vote90 ~ prtystr + democ + linexp90 + lchexp90), vote2)
lm(@formula(vote90 ~ prtystr + democ + linexp90 + lchexp90 + vote88), vote2)

# Problem 3

meap93 = get_dataset("meap93")
math10_multi1 = lm(@formula(math10 ~ lexpend + lenroll), meap93)
math10_multi2 = lm(@formula(math10 ~ lexpend + lenroll + lnchprg), meap93)

# Problem C1

ceosal1 = get_dataset("ceosal1")
lsalary_multi = lm(@formula(lsalary ~ lsales + roe + identity(ros<0)), ceosal1)
ceosal1.pred = predict(lsalary_multi)
lsalary_multi2 = lm(@formula(lsalary ~ lsales + roe + identity(ros<0) + pred^2 + pred^3), ceosal1)
ftest(lsalary_multi.model, lsalary_multi2.model)
lm_waldtest(lsalary_multi2, [0.0 0.0 0.0 0.0 1.0 0.0
                             0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0])
lm_waldtest(lsalary_multi2, [0.0 0.0 0.0 0.0 1.0 0.0
                             0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0], :HC0)

# Problem C3

jtrain = get_dataset("jtrain")
jtrain88 = @rsubset(jtrain, :year == 1988)
lscrap_grant = lm(@formula(lscrap ~ grant), jtrain88)
nobs(lscrap_grant)
r2(lscrap_grant)

jtrain88.lscrap87 = jtrain.lscrap[jtrain.year .== 1987]
lscrap_grant2 = lm(@formula(lscrap ~ grant + lscrap87), jtrain88)
nobs(lscrap_grant2)
r2(lscrap_grant2)
lm_ttest(lscrap_grant2, :HC0)

lscrap_grant3 = lm(@formula(lscrap-lscrap87 ~ grant + lscrap87), jtrain88)
lm_ttest(lscrap_grant3, :HC0)

# Problem C6

meap93 = get_dataset("meap93")
lsalary_multi1 = lm(@formula(lsalary ~ identity(benefits/salary) + lenroll + lstaff + droprate + gradrate), meap93)
nobs(lsalary_multi1)
r2(lsalary_multi1)
lsalary_multi2 = lm(@formula(lsalary ~ identity(benefits/salary) + lenroll + lstaff + droprate + gradrate),
                    @rsubset(meap93, :benefits / :salary > 0.01))
nobs(lsalary_multi2)
r2(lsalary_multi2)

# Problem C7

loanapp = get_dataset("loanapp")
count(>(40.0), loanapp.obrat)

lm(@formula(approve ~ white + hrat + obrat + loanprc + unem + male + married + dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr),
            @rsubset(loanapp, :obrat <= 40.0))

# Problem C8

twoyear = get_dataset("twoyear")

describe(twoyear, :mean, :std, cols=:stotal)
jc_stotal = lm(@formula(jc ~ stotal), twoyear)
univ_stotal = lm(@formula(univ ~ stotal), twoyear)

lwage_multi1 = lm(@formula(lwage ~ jc + univ + exper + stotal), twoyear)
lwage_multi2 = lm(@formula(lwage ~ identity(jc + univ) + univ + exper + stotal), twoyear)
lwage_multi3 = lm(@formula(lwage ~ identity(jc + univ) + univ + exper + stotal + stotal^2), twoyear)
lwage_multi4 = lm(@formula(lwage ~ identity(jc + univ) + univ + exper + stotal + stotal&jc + stotal&univ), twoyear)
lm_waldtest(lwage_multi4, [0.0 0.0 0.0 0.0 0.0 1.0 0.0
                           0.0 0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0])

# Problem C9

k401ksubs = get_dataset("k401ksubs")

nettfa_multi = lm(@formula(nettfa ~ inc + inc^2 + age + age^2 + male + e401k), k401ksubs)

k401ksubs.u = residuals(nettfa_multi)
u2_multi = lm(@formula(u^2 ~ inc + inc^2 + age + age^2 + male + e401k), k401ksubs)
r2(u2_multi)
nobs(u2_multi)
ftest(u2_multi.model)
LM = r2(u2_multi) * nobs(u2_multi)
ccdf(Chisq(6), LM)

lad(nettfa_multi)

# Problem C10

jtrain2 = get_dataset("jtrain2")
jtrain3 = get_dataset("jtrain3")
mean(jtrain2.train), mean(jtrain3.train)

re78_train1 = lm(@formula(re78 ~ train), jtrain2)
re78_multi1 = lm(@formula(re78 ~ train + re74 + re75 + educ + age + black + hisp), jtrain2)

re78_train2 = lm(@formula(re78 ~ train), jtrain3)
re78_multi2 = lm(@formula(re78 ~ train + re74 + re75 + educ + age + black + hisp), jtrain3)

@rtransform!(jtrain2, :avgre = (:re74 + :re75) / 2)
describe(jtrain2, :all, cols=:avgre)
describe(jtrain3, :all, cols=:avgre)

re78_multi3 = lm(@formula(re78 ~ train + re74 + re75 + educ + age + black + hisp),
                 @rsubset(jtrain2, :avgre < 10.0))
re78_multi4 = lm(@formula(re78 ~ train + re74 + re75 + educ + age + black + hisp),
                 @rsubset(jtrain3, :avgre < 10.0))

# Problem C11

murder = get_dataset("murder")
transform!(groupby(murder, :state), :mrdrte => lag)

mrdrte_exec_unem = lm(@formula(mrdrte ~ exec + unem), murder)
sort(unstack(murder, :state, :year, :exec), "93", rev=true)
murder.TX = murder.state .== "TX"
mrdrte_multi1 = lm(@formula(mrdrte ~ exec + unem + TX), murder)
mrdrte_multi2 = lm(@formula(mrdrte ~ exec + unem + mrdrte_lag), murder)
mrdrte_multi3 = lm(@formula(mrdrte ~ exec + unem + mrdrte_lag), @rsubset(murder, !:TX))

# Problem C12

elem94_95 = get_dataset("elem94_95")

lavgsal_multi1 = lm(@formula(lavgsal ~ bs + lenrol + lstaff + lunch), elem94_95)
lm_ttest(lavgsal_multi1, :HC3)

lavgsal_multi2 = lm(@formula(lavgsal ~ bs + lenrol + lstaff + lunch),
                    @rsubset(elem94_95, :bs <= 0.5))
lm_ttest(lavgsal_multi2, :HC3)

findall(elem94_95.bs .> 0.5)
for ind in findall(elem94_95.bs .> 0.5)
    elem94_95[!, "d$ind"] = axes(elem94_95, 1) .== ind
end

lavgsal_multi3 = lm(@formula(lavgsal ~ bs + lenrol + lstaff + lunch + d68 + d1127 + d1508 + d1670), elem94_95)

lm(@formula(lavgsal ~ bs + lenrol + lstaff + lunch),
   @rsubset(elem94_95, !:d68))
lm(@formula(lavgsal ~ bs + lenrol + lstaff + lunch),
   @rsubset(elem94_95, !:d1127))
lavgsal_multi4 = lm(@formula(lavgsal ~ bs + lenrol + lstaff + lunch),
                    @rsubset(elem94_95, !:d1508))
lm(@formula(lavgsal ~ bs + lenrol + lstaff + lunch),
   @rsubset(elem94_95, !:d1670))

lad(lavgsal_multi1)
lad(lavgsal_multi4)

# Problem C13

ceosal2 = get_dataset("ceosal2")

lsalary_multi1 = lm(@formula(lsalary ~ lsales + lmktval + ceoten + ceoten^2), ceosal2)
lm_ttest(lsalary_multi1, :HC3)
studentized_residuals(lsalary_multi1)
sum(abs.(studentized_residuals(lsalary_multi1)) .> 1.96)
mean(abs.(studentized_residuals(lsalary_multi1)) .> 1.96)

ceosal2.nonextreme = abs.(studentized_residuals(lsalary_multi1)) .<= 1.96
lsalary_multi2 = lm(@formula(lsalary ~ lsales + lmktval + ceoten + ceoten^2),
                    @subset(ceosal2, :nonextreme))
lad(lsalary_multi1)


# Problem C14

econmath = get_dataset("econmath")
count(ismissing, econmath.act)
mean(ismissing, econmath.act)
econmath.actmiss = ismissing.(econmath.act)
econmath.act0 = coalesce.(econmath.act, zero(eltype(econmath.act)))
describe(econmath, cols=[:act, :act0])

score_act = lm(@formula(score ~ act), econmath)
lm_ttest(score_act, :HC3)

score_act0 = lm(@formula(score ~ act0), econmath)
lm_ttest(score_act0, :HC3)

score_act0_actmiss = lm(@formula(score ~ act0 + actmiss), econmath)
lm_ttest(score_act0_actmiss, :HC3)

score_act2 = lm(@formula(score ~ act + colgpa), econmath)
score_act0_actmiss2 = lm(@formula(score ~ act0 + actmiss + colgpa), econmath)
