include("init_example.jl")

# Example 4.1

wage1 = get_dataset("wage1")
lwage_multi = lm(@formula(lwage ~ educ + exper + tenure), wage1)
nobs(lwage_multi)
r2(lwage_multi)
quantile(Normal(), [0.95, 0.99])

# Example 4.2 and Problem 12

meap93 = get_dataset("meap93")
math10_multi1 = lm(@formula(math10 ~ totcomp + staff + enroll), meap93)
nobs(math10_multi1)
r2(math10_multi1)

math10_multi2 = lm(@formula(math10 ~ ltotcomp + lstaff + lenroll), meap93)
nobs(math10_multi2)
r2(math10_multi2)

math10_multi3 = lm(@formula(math10 ~ lexpend), meap93)
nobs(math10_multi3)
r2(math10_multi3)

math10_multi4 = lm(@formula(math10 ~ lexpend + lenroll + lnchprg), meap93)
nobs(math10_multi4)
r2(math10_multi4)

# Example 4.3 and Problem 5

gpa1 = get_dataset("gpa1")
colGPA_multi = lm(@formula(colGPA ~ hsGPA + ACT + skipped), gpa1)
nobs(colGPA_multi)
r2(colGPA_multi)

ct_colGPA_multi = DataFrame(coeftable(colGPA_multi))
(ct_colGPA_multi[2, "Coef."] - 0.4) / ct_colGPA_multi[2, "Std. Error"]
(ct_colGPA_multi[2, "Coef."] - 1) / ct_colGPA_multi[2, "Std. Error"]
quantile(TDist(137), 0.975)

# Example 4.4

campus = get_dataset("campus")
lcrime_lenroll = lm(@formula(lcrime ~ lenroll), campus)
nobs(lcrime_lenroll)
r2(lcrime_lenroll)

ct_lcrime_lenroll = DataFrame(coeftable(lcrime_lenroll))
(ct_lcrime_lenroll[2, "Coef."] - 1) / ct_lcrime_lenroll[2, "Std. Error"]
quantile(TDist(95), [0.95, 0.99])

# Example 4.5

hprice2 = get_dataset("hprice2")
lprice_multi = lm(@formula(lprice ~ lnox + log(dist) + rooms + stratio), hprice2)
nobs(lprice_multi)
r2(lprice_multi)

ct_lprice_multi = DataFrame(coeftable(lprice_multi))
(ct_lprice_multi[2, "Coef."] + 1) / ct_lprice_multi[2, "Std. Error"]

# Example 4.6

k401k = get_dataset("k401k")
prate_multi = lm(@formula(prate ~ mrate + age + totemp), k401k)
nobs(prate_multi)
r2(prate_multi)

# Example 4.7 and Problem 7

jtrain = get_dataset("jtrain")
jtrain_sub = @rsubset(jtrain, :year == 1987, :union == 0) 
lscrap_multi1 = lm(@formula(lscrap ~ hrsemp + lsales + lemploy), jtrain_sub)
nobs(lscrap_multi1)
r2(lscrap_multi1)

jtrain_sub2 = @rsubset(jtrain, :year == 1987)
lscrap_multi2 = lm(@formula(lscrap ~ hrsemp + lsales + lemploy), jtrain_sub2)
nobs(lscrap_multi2)
r2(lscrap_multi2)

lscrap_multi3 = lm(@formula(lscrap ~ hrsemp + log(sales/employ) + lemploy), jtrain_sub2)
nobs(lscrap_multi3)
r2(lscrap_multi3)

ct_lscrap_multi3 = DataFrame(coeftable(lscrap_multi3))
(ct_lscrap_multi3[3, "Coef."] + 1) / ct_lscrap_multi3[3, "Std. Error"]
quantile(TDist(39), 0.99)

# Example 4.8

rdchem = get_dataset("rdchem")
lrd_multi = lm(@formula(lrd ~ lsales + profmarg), rdchem)
nobs(lrd_multi)
r2(lrd_multi)

# Section 4.4 and Problem C7

twoyear = get_dataset("twoyear")

lwage_multi1 = lm(@formula(lwage ~ jc + univ + exper), twoyear)
nobs(lwage_multi1)
r2(lwage_multi1)

lwage_multi2 = lm(@formula(lwage ~ jc + identity(jc + univ) + exper), twoyear)
nobs(lwage_multi2)
r2(lwage_multi2)

describe(twoyear.phsrank)

lwage_multi3 = lm(@formula(lwage ~ jc + identity(jc + univ) + exper + phsrank), twoyear)
nobs(lwage_multi3)
r2(lwage_multi3)

lwage_multi4 = lm(@formula(lwage ~ jc + univ + exper + id), twoyear)

# Section 4.5 and Problem C5

mlb1 = get_dataset("mlb1")

lsalary_multi1 = lm(@formula(lsalary ~ years + gamesyr + bavg + hrunsyr + rbisyr), mlb1)
nobs(lsalary_multi1)
r2(lsalary_multi1)
deviance(lsalary_multi1)

lsalary_multi2 = lm(@formula(lsalary ~ years + gamesyr), mlb1)
nobs(lsalary_multi2)
r2(lsalary_multi2)

deviance(lsalary_multi2)
(deviance(lsalary_multi2) / deviance(lsalary_multi1) - 1) * 347 / 3
quantile(FDist(3, 347), 0.99)

lsalary_multi3 = lm(@formula(lsalary ~ years + gamesyr + bavg + hrunsyr), mlb1)
nobs(lsalary_multi3)
r2(lsalary_multi3)

lsalary_multi4 = lm(@formula(lsalary ~ years + gamesyr + bavg + hrunsyr + runsyr + fldperc + sbasesyr), mlb1)
nobs(lsalary_multi4)
r2(lsalary_multi4)

lsalary_multi5 = lm(@formula(lsalary ~ years + gamesyr + hrunsyr + runsyr), mlb1)
nobs(lsalary_multi5)
r2(lsalary_multi5)

ftest(lsalary_multi5.model, lsalary_multi4.model)

# Example 4.9 and Problem C4

bwght = get_dataset("bwght")

# this will error
bwght_multi1 = lm(@formula(bwght ~ cigs + parity + faminc + motheduc + fatheduc), bwght)
nobs(bwght_multi1)
bwght_multi2 = lm(@formula(bwght ~ cigs + parity + faminc), bwght) # needed for C4
nobs(bwght_multi2)
r2(bwght_multi2)
ftest(bwght_multi2.model, bwght_multi1.model)

# need to ensure the same data is used in both models
bwght2 = select(bwght, :bwght, :cigs, :parity, :faminc, :motheduc, :fatheduc)
dropmissing!(bwght2)
bwght_multi3 = lm(@formula(bwght ~ cigs + parity + faminc + motheduc + fatheduc), bwght2)
nobs(bwght_multi3)
bwght_multi4 = lm(@formula(bwght ~ cigs + parity + faminc), bwght2)
nobs(bwght_multi4)
r2(bwght_multi4)
ftest(bwght_multi4.model, bwght_multi3.model)

# Section 4.5f and Problem 6

hprice1 = get_dataset("hprice1")
lprice_model = lm(@formula(lprice ~ lassess + llotsize + lsqrft + bdrms), hprice1)
nobs(lprice_model)
deviance(lprice_model)

lprice_ref = lm(@formula(identity(lprice - lassess) ~ 1), hprice1)
nobs(lprice_ref)
deviance(lprice_ref)

(deviance(lprice_ref) / deviance(lprice_model) - 1) * 83 / 4
quantile(FDist(4, 83), 0.95)

p6_model1 = lm(@formula(identity(price - assess) ~ assess), hprice1)
p6_model2 = lm(@formula(identity(price - assess) ~ assess + lotsize + sqrft + bdrms), hprice1)

ftest(p6_model1.model)
ftest(p6_model1.model, p6_model2.model)

# Example 4.10

meap93 = get_dataset("meap93")
lsalary_multi1 = lm(@formula(lsalary ~ identity(benefits/salary)), meap93)
nobs(lsalary_multi1)
r2(lsalary_multi1)

lsalary_multi2 = lm(@formula(lsalary ~ identity(benefits/salary) + lenroll + lstaff), meap93)
nobs(lsalary_multi2)
r2(lsalary_multi2)

lsalary_multi3 = lm(@formula(lsalary ~ identity(benefits/salary) + lenroll + lstaff + droprate + gradrate), meap93)
nobs(lsalary_multi3)
r2(lsalary_multi3)

ct_lsalary_multi1 = DataFrame(coeftable(lsalary_multi1))
(ct_lsalary_multi1[2, "Coef."] + 1) / ct_lsalary_multi1[2, "Std. Error"]

ct_lsalary_multi2 = DataFrame(coeftable(lsalary_multi2))
(ct_lsalary_multi2[2, "Coef."] + 1) / ct_lsalary_multi2[2, "Std. Error"]

ftest(lsalary_multi2.model, lsalary_multi3.model)

# Example 4.11

jtrain98 = get_dataset("jtrain98")
earn98_train = lm(@formula(earn98 ~ train), jtrain98)
nobs(earn98_train)
r2(earn98_train)

earn98_multi1 = lm(@formula(earn98 ~ train + earn96 + educ + age + married), jtrain98)
nobs(earn98_multi1)
r2(earn98_multi1)

earn98_multi2 = lm(@formula(earn98 ~ train + earn96 + educ + age + married + unem96), jtrain98)
nobs(earn98_multi2)
r2(earn98_multi2)

cor(jtrain98.earn96, jtrain98.unem96)

# Problem 2

ceosal1 = get_dataset("ceosal1")
lsalary_multi = lm(@formula(lsalary ~ lsales + roe + ros), ceosal1)
nobs(lsalary_multi)
r2(lsalary_multi)

# Problem 3

rdchem = get_dataset("rdchem")
rdintens_multi = lm(@formula(rdintens ~ lsales + profmarg), rdchem)
nobs(rdintens_multi)
r2(rdintens_multi)

confint(rdintens_multi; level=0.9)
confint(rdintens_multi; level=0.8)

# Problem 4

rental = get_dataset("rental")
rental90 = @rsubset(rental, :year == 90)
lrent_multi = lm(@formula(lrent ~ lpop + lavginc + pctstu), rental90)
nobs(lrent_multi)
r2(lrent_multi)

# Problem 9

sleep75 = get_dataset("sleep75")
sleep_multi1 = lm(@formula(sleep ~ totwrk + educ + age), sleep75)
nobs(sleep_multi1)
r2(sleep_multi1)

sleep_multi2 = lm(@formula(sleep ~ totwrk), sleep75)
nobs(sleep_multi2)
r2(sleep_multi2)
ftest(sleep_multi2.model, sleep_multi1.model)

# Problem 10

ret = get_dataset("return")
ret_multi1 = lm(@formula(_return ~ dkr + eps + netinc + salary), ret)
nobs(ret_multi1)
r2(ret_multi1)

ret_multi2 = lm(@formula(_return ~ dkr + eps + lnetinc + lsalary), ret)
nobs(ret_multi2)
r2(ret_multi2)

# Problem 11

ceosal2 = get_dataset("ceosal2")

lsalary_multi1 = lm(@formula(lsalary ~ lsales), ceosal2)
nobs(lsalary_multi1)
r2(lsalary_multi1)

lsalary_multi2 = lm(@formula(lsalary ~ lsales + lmktval + profmarg), ceosal2)
nobs(lsalary_multi2)
r2(lsalary_multi2)

lsalary_multi3 = lm(@formula(lsalary ~ lsales + lmktval + profmarg + ceoten + comten), ceosal2)
nobs(lsalary_multi3)
r2(lsalary_multi3)

# Problem 13

meapsingle = get_dataset("meapsingle")

math4_multi1 = lm(@formula(math4 ~ pctsgle), meapsingle)
nobs(math4_multi1)
r2(math4_multi1)

math4_multi2 = lm(@formula(math4 ~ pctsgle + free), meapsingle)
nobs(math4_multi2)
r2(math4_multi2)

math4_multi3 = lm(@formula(math4 ~ pctsgle + free + lmedinc + lexppp), meapsingle)
nobs(math4_multi3)
r2(math4_multi3)

math4_multi4 = lm(@formula(math4 ~ pctsgle + free + lexppp), meapsingle)
nobs(math4_multi4)
r2(math4_multi4)

# Problem C1

vote1 = get_dataset("vote1")

voteA_multi1 = lm(@formula(voteA ~ lexpendA + lexpendB + prtystrA), vote1)
nobs(voteA_multi1)
r2(voteA_multi1)

voteA_multi2 = lm(@formula(voteA ~ identity(lexpendA - lexpendB) + lexpendB + prtystrA), vote1)
nobs(voteA_multi2)
r2(voteA_multi2)

# Problem C2

lawsch85 = get_dataset("lawsch85")

lsalary_multi = lm(@formula(lsalary ~ LSAT + GPA + llibvol + lcost + rank), lawsch85)
nobs(lsalary_multi)
r2(lsalary_multi)

lawsch85_ref = dropmissing!(select(lawsch85, :lsalary, :LSAT, :GPA, :llibvol, :lcost, :rank))
lsalary_multi_ref = lm(@formula(lsalary ~ llibvol + lcost + rank), lawsch85_ref)
nobs(lsalary_multi_ref)
r2(lsalary_multi_ref)

ftest(lsalary_multi_ref.model, lsalary_multi.model)


lawsch85_ref2 = dropmissing!(select(lawsch85, :lsalary, :LSAT, :GPA, :llibvol, :lcost, :rank, :clsize, :faculty))
lsalary_multi2 = lm(@formula(lsalary ~ LSAT + GPA + llibvol + lcost + rank + clsize + faculty), lawsch85_ref2)
nobs(lsalary_multi2)
r2(lsalary_multi2)

lsalary_multi2_ref = lm(@formula(lsalary ~ LSAT + GPA + llibvol + lcost + rank), lawsch85_ref2)
nobs(lsalary_multi2_ref)
r2(lsalary_multi2_ref)

ftest(lsalary_multi2_ref.model, lsalary_multi2.model)

# Problem C3

hprice1 = get_dataset("hprice1")
price_sqrft_bdrms = lm(@formula(price ~ sqrft + bdrms), hprice1)
nobs(price_sqrft_bdrms)
r2(price_sqrft_bdrms)

[150 1] * coef(price_sqrft_bdrms)[2:3]

price_sqrft_bdrms_2 = lm(@formula(price ~ bdrms + identity(sqrft - 150.0 * bdrms)), hprice1)
nobs(price_sqrft_bdrms_2)
r2(price_sqrft_bdrms_2)

# Problem C6

wage2 = get_dataset("wage2")
lwage_multi_1 = lm(@formula(lwage ~ educ + exper + tenure), wage2)
lwage_multi_2 = lm(@formula(lwage ~ educ + exper + identity(exper + tenure)), wage2)

# Problem C8

k401ksubs = get_dataset("k401ksubs")
k401ksubs_s = @rsubset(k401ksubs, :fsize == 1)
nrow(k401ksubs_s)

nettfa_multi1 = lm(@formula(nettfa ~ inc + age), k401ksubs_s)
nobs(nettfa_multi1)
r2(nettfa_multi1)

nettfa_multi2 = lm(@formula(identity(nettfa-inc) ~ inc  + age), k401ksubs_s)
nobs(nettfa_multi2)
r2(nettfa_multi2)

nettfa_multi3 = lm(@formula(nettfa ~ inc), k401ksubs_s)
nobs(nettfa_multi3)
r2(nettfa_multi3)

cor(k401ksubs_s.inc, k401ksubs_s.age)
cor(k401ksubs_s.nettfa, k401ksubs_s.age)
cor(k401ksubs_s.nettfa, k401ksubs_s.inc)

# Problem C9

discrim = get_dataset("discrim")

lpsoda_multi1 = lm(@formula(lpsoda ~ prpblck + lincome + prppov), discrim)
nobs(lpsoda_multi1)
r2(lpsoda_multi1)

cor(collect.(skipmissings(discrim.lincome, discrim.prppov))...)

lpsoda_multi2 = lm(@formula(lpsoda ~ prpblck + lincome + prppov + hseval), discrim)
nobs(lpsoda_multi2)
r2(lpsoda_multi2)

lpsoda_multi3 = lm(@formula(lpsoda ~ prpblck + hseval), discrim)
nobs(lpsoda_multi3)
r2(lpsoda_multi3)

ftest(lpsoda_multi3.model, lpsoda_multi2.model)

# Problem C10

elem94_95 = get_dataset("elem94_95")

lavgsal_bs = lm(@formula(lavgsal ~ bs), elem94_95)
nobs(lavgsal_bs)
r2(lavgsal_bs)

lm(@formula(identity(lavgsal + bs) ~ bs), elem94_95)

ct_lavgsal_bs = DataFrame(coeftable(lavgsal_bs))
(ct_lavgsal_bs[2, "Coef."] + 1) / ct_lavgsal_bs[2, "Std. Error"]

lavgsal_multi1 = lm(@formula(lavgsal ~ bs + lenrol + lstaff), elem94_95)
nobs(lavgsal_multi1)
r2(lavgsal_multi1)


lavgsal_multi2 = lm(@formula(lavgsal ~ bs + lenrol + lstaff + lunch), elem94_95)
nobs(lavgsal_multi2)
r2(lavgsal_multi2)

# Problem C11

htv = get_dataset("htv")

educ_multi1 = lm(@formula(educ ~ motheduc + fatheduc + abil + abil^2), htv)
nobs(educ_multi1)
r2(educ_multi1)

educ_multi2 = lm(@formula(educ ~ motheduc + identity(motheduc+fatheduc) + abil + abil^2), htv)

educ_multi3 = lm(@formula(educ ~ motheduc + fatheduc + abil + abil^2 + tuit17 + tuit18), htv)
nobs(educ_multi3)
r2(educ_multi3)

ftest(educ_multi1.model, educ_multi3.model)
cor(htv.tuit17, htv.tuit18)

educ_multi4 = lm(@formula(educ ~ motheduc + fatheduc + abil + abil^2 + identity((tuit17 + tuit18) / 2)), htv)
nobs(educ_multi4)
r2(educ_multi4)

# Problem C12

econmath = get_dataset("econmath")
score_multi1 = lm(@formula(score ~ colgpa + hsgpa + actmth + acteng), econmath)
nobs(score_multi1)
r2(score_multi1)

score_multi2 = lm(@formula(score ~ colgpa + hsgpa + actmth + identity(actmth + acteng)), econmath)

# Problem C13

gpa1 = get_dataset("gpa1")

colGPA_multi1 = lm(@formula(colGPA ~ PC + hsGPA + ACT), gpa1)
nobs(colGPA_multi1)
r2(colGPA_multi1)

colGPA_multi2 = lm(@formula(colGPA ~ PC + hsGPA + ACT + fathcoll + mothcoll), gpa1)
nobs(colGPA_multi2)
r2(colGPA_multi2)

ftest(colGPA_multi1.model, colGPA_multi2.model)

