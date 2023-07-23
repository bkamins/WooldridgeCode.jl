include("init_example.jl")

# Table 6.1

bwght = get_dataset("bwght")
bwght_multi1 = lm(@formula(bwght ~ cigs + faminc), bwght)
bwght_multi2 = lm(@formula(bwght/16 ~ cigs + faminc), bwght)
bwght_multi3 = lm(@formula(bwght ~ packs + faminc), bwght)
r2(bwght_multi1)
r2(bwght_multi2)
r2(bwght_multi3)

bwght_multi4 = lm(@formula(bwght ~ cigs + identity(1000 * faminc)), bwght)

# Example 6.1 and Section 6-2a and Example 6.2

hprice2 = get_dataset("hprice2")
price_multi1 = lm(@formula(price ~ nox + crime + rooms + dist + stratio),
                  hprice2)
price_multi1 = lm(@formula(price ~ nox + crime + rooms + dist + stratio),
                  mapcols(zscore, hprice2))

lprice_multi1 = lm(@formula(lprice ~ lnox + rooms), hprice2)

lprice_multi2 = lm(@formula(lprice ~ lnox + log(dist) + rooms + rooms^2 + stratio),
                   hprice2)

# Section 6-2b

wage1 = get_dataset("wage1")
wage_multi = lm(@formula(wage ~ exper + exper^2), wage1)
β₀, β₁, β₂ = coef(wage_multi)
max_exper = [-β₁ / (2 * β₂)]
max_wage = predict(wage_multi, DataFrame(exper=max_exper))

plot(x -> β₀ + β₁ * x + β₂ * x^2;
     xlim=extrema(wage1.exper),
     xlabel="exper", ylabel="predicted wage", label=false)
scatter!(max_exper, max_wage, label=false)

# Example 6.3 and Problem 6 and C7

attend = get_dataset("attend")
stndfnl_multi_1 = lm(@formula(stndfnl ~ ACT + ACT^2 + priGPA*atndrte + priGPA^2), attend)
r2(stndfnl_multi_1)

stndfnl_multi_2 = lm(@formula(stndfnl ~ ACT + ACT^2 + priGPA*atndrte + priGPA^2 +
                              atndrte^2 + ACT&atndrte), attend)
r2(stndfnl_multi_2)

ftest(stndfnl_multi_1.model, stndfnl_multi_2.model)

β₂, β₄, β₆ = coef(stndfnl_multi_1)[[4, 6, 7]]
β₂ + 2 * β₄ * 2.59 + β₆ * 82

stndfnl_multi_3 = lm(@formula(stndfnl ~ atndrte + priGPA + ACT + (priGPA-2.59)^2 +
                              ACT^2 + priGPA&(atndrte-82)), attend)
r2(stndfnl_multi_3)

# Section 6-3b and Problem 3

rdchem = get_dataset("rdchem")
rdintens_multi1 = lm(@formula(rdintens ~ lsales), rdchem)
r2(rdintens_multi1)
adjr2(rdintens_multi1)
# note that we need to set dropcollinear=false as the columns are almost collinear
rdintens_multi2 = lm(@formula(rdintens ~ sales + sales^2), rdchem, dropcollinear=false)
r2(rdintens_multi2)
adjr2(rdintens_multi2)

plot(x -> [1.0, x, x^2]' * coef(rdintens_multi2); xlim=extrema(rdchem.sales),
     xlabel="sales", ylabel="rdintens", label=false)
x = -coef(rdintens_multi2)[2] / (2 * coef(rdintens_multi2)[3])
scatter!([x], predict(rdintens_multi2, DataFrame(sales=x)), label=false)

# Example 6.4 and Problem 1

ceosal1 = get_dataset("ceosal1")
salary_multi = lm(@formula(salary ~ sales + roe), ceosal1)
r2(salary_multi)
adjr2(salary_multi)
lsalary_multi = lm(@formula(lsalary ~ lsales + roe), ceosal1)
r2(lsalary_multi)
adjr2(lsalary_multi)

lsalary_multi = lm(@formula(lsalary ~ lsales + roe + roe^2), ceosal1)

# Example 6.5 and 6.6

gpa2 = get_dataset("gpa2")
colgpa_multi = lm(@formula(colgpa ~ sat + hsperc + hsize + hsize^2), gpa2)
nobs(colgpa_multi)
r2(colgpa_multi)
adjr2(colgpa_multi)

predict(colgpa_multi, DataFrame(sat=1200, hsperc=30, hsize=5))
predict(colgpa_multi, DataFrame(sat=1200, hsperc=30, hsize=5);
        interval=:confidence)
predict(colgpa_multi, DataFrame(sat=1200, hsperc=30, hsize=5);
        interval=:prediction)

colgpa_multi_0 = lm(@formula(colgpa ~
                             identity(sat - 1200) +
                             identity(hsperc - 30 ) +
                             identity(hsize - 5) +
                             identity(hsize^2 - 25)), gpa2)
nobs(colgpa_multi_0)
r2(colgpa_multi_0)
adjr2(colgpa_multi_0)

predict(colgpa_multi_0, DataFrame(sat=1200, hsperc=30, hsize=5))
predict(colgpa_multi_0, DataFrame(sat=1200, hsperc=30, hsize=5);
        interval=:confidence)
predict(colgpa_multi_0, DataFrame(sat=1200, hsperc=30, hsize=5);
        interval=:prediction)

# Example 6.7 and 6.8 and Problem 9

ceosal2 = get_dataset("ceosal2")
lsalary_multi = lm(@formula(lsalary ~ lsales + lmktval + ceoten), ceosal2)
p = predict(lsalary_multi, DataFrame(lsales=log(5000), lmktval=log(10000), ceoten=10))

α1 = mean(exp, residuals(lsalary_multi))
exp(only(p)) * α1

y = ceosal2.salary
m = exp.(predict(lsalary_multi))
α2 = (m' * y) / (m' * m)
exp(only(p)) * α2

cor(m, y) ^ 2
1 - sum((α1*m - y) .^ 2) / sum((y .- mean(y)) .^ 2)
1 - sum((α2*m - y) .^ 2) / sum((y .- mean(y)) .^ 2)

salary_multi = lm(@formula(salary ~ sales + mktval + ceoten), ceosal2)
r2(salary_multi)

σ²_hat = ser(lsalary_multi)^2
exp.(p .+ [-1.96, 1.96] * σ²_hat)
exp(σ²_hat/2)*exp(only(p))

# Problem 4

wage2 = get_dataset("wage2")
wage2.pareduc = wage2.meduc + wage2.feduc
lwage_multi1 = lm(@formula(lwage ~ educ + educ&pareduc + exper + tenure), wage2)
nobs(lwage_multi1)
r2(lwage_multi1)

lwage_multi2 = lm(@formula(lwage ~ educ*pareduc + exper + tenure), wage2)
nobs(lwage_multi2)
r2(lwage_multi2)

# Problem 7

k401k = get_dataset("k401k")
prate_multi1 = lm(@formula(prate ~ mrate + age + totemp), k401k)
nobs(prate_multi1)
r2(prate_multi1)
adjr2(prate_multi1)

prate_multi2 = lm(@formula(prate ~ mrate + age + ltotemp), k401k)
nobs(prate_multi2)
r2(prate_multi2)
adjr2(prate_multi2)

# division by 1000 due to numerical stability issues
# TODO: in GLM.jl 2.0 it will not be needed; use method=:qr
prate_multi3 = lm(@formula(prate ~ mrate + age + identity(totemp/1000) +
                           identity((totemp/1000)^2)), k401k)
nobs(prate_multi3)
r2(prate_multi3)
adjr2(prate_multi3)

# Problem 10

meapsingle = get_dataset("meapsingle")
math4_multi1 = lm(@formula(math4 ~ lexppp + free + lmedinc + pctsgle), meapsingle)
nobs(math4_multi1)
r2(math4_multi1)
adjr2(math4_multi1)
math4_multi2 = lm(@formula(math4 ~ lexppp + free + lmedinc + pctsgle + read4), meapsingle)
nobs(math4_multi2)
r2(math4_multi2)
adjr2(math4_multi2)

# Problem C1

kielmc = get_dataset("kielmc")
kielmc1981 = @rsubset(kielmc, :year == 1981)
lprice_ldist = lm(@formula(lprice ~ ldist), kielmc1981)
lprice_multi1 = lm(@formula(lprice ~ ldist + lintst + larea + lland + rooms + baths + age),
                   kielmc1981)
lprice_multi2 = lm(@formula(lprice ~ ldist + lintst + lintst^2 + larea + lland + rooms + baths + age),
                   kielmc1981)

lprice_multi3 = lm(@formula(lprice ~ ldist + ldist^2 + lintst + lintst^2 + larea + lland + rooms + baths + age),
                   kielmc1981)

# Problem C2

wage1 = get_dataset("wage1")
lwage_multi = lm(@formula(lwage ~ educ + exper + exper^2), wage1)
β₂, β₃ = coef(lwage_multi)[3:4]
100 * (β₂ + 2 * β₃ * 4)
100 * (β₂ + 2 * β₃ * 19)
-β₂ / (2 * β₃)
count(>(-β₂ / (2 * β₃)),wage1.exper)
mean(>(-β₂ / (2 * β₃)),wage1.exper)

# Problem C3

wage2 = get_dataset("wage2")
lwage_multi1 = lm(@formula(lwage ~ educ * exper), wage2)
lwage_multi2 = lm(@formula(lwage ~ educ * (exper - 10)), wage2)

# Problem C4

gpa2 = get_dataset("gpa2")
sat_multi = lm(@formula(sat ~ hsize + hsize^2), gpa2)
- coef(sat_multi)[2] / (2 * coef(sat_multi)[3])

lsat_multi = lm(@formula(log(sat) ~ hsize + hsize^2), gpa2)
- coef(lsat_multi)[2] / (2 * coef(lsat_multi)[3])

# Problem C5 and C8

hprice1 = get_dataset("hprice1")
lprice_multi = lm(@formula(lprice ~ llotsize + lsqrft + bdrms), hprice1)

p = predict(lprice_multi, DataFrame(llotsize=log(20000), lsqrft=log(2500), bdrms=10))

α1 = mean(exp, residuals(lprice_multi))
exp(only(p)) * α1

y = hprice1.price
m = exp.(predict(lprice_multi))
α2 = (m' * y) / (m' * m)
exp(only(p)) * α2

cor(m, y) ^ 2
1 - sum((α1*m - y) .^ 2) / sum((y .- mean(y)) .^ 2)
1 - sum((α2*m - y) .^ 2) / sum((y .- mean(y)) .^ 2)

price_multi = lm(@formula(price ~ lotsize + sqrft + bdrms), hprice1)
nobs(price_multi)
r2(price_multi)
ser(price_multi)^2

predict(price_multi, DataFrame(lotsize=10000, sqrft=2300, bdrms=4);
        interval=:confidence)
predict(price_multi, DataFrame(lotsize=10000, sqrft=2300, bdrms=4);
        interval=:prediction)

# Problem C6

vote1 = get_dataset("vote1")
voteA_multi1 = lm(@formula(voteA ~ prtystrA + expendA*expendB), vote1)
mean(vote1.expendA)
diff(predict(voteA_multi1, DataFrame(prtystrA=0, expendA=300, expendB=[0, 100])))
diff(predict(voteA_multi1, DataFrame(prtystrA=0, expendA=[0, 100], expendB=100)))

voteA_multi2 = lm(@formula(voteA ~ prtystrA + expendA + expendB + shareA), vote1)

voteA_multi3 = lm(@formula(voteA ~ prtystrA + expendA + expendB +
                           identity(expendA / (expendA + expendB))), vote1)
β₄, β₅ = coef(voteA_multi3)[4:5]
β₄ - β₅ * 300 / (300 + 0)^2

# Problem C9

nbasal = get_dataset("nbasal")

points_multi1 = lm(@formula(points ~ exper + exper^2 + age + coll), nbasal)
- coef(points_multi1)[2] / (2 * coef(points_multi1)[3])

points_multi2 = lm(@formula(points ~ exper + exper^2 + age + age^2 + coll), nbasal)

lwage_multi1 = lm(@formula(lwage ~ points + exper + exper^2 + age + coll), nbasal)
lwage_multi2 = lm(@formula(lwage ~ points + exper + exper^2), nbasal)
ftest(lwage_multi2.model, lwage_multi1.model)

# Problem C10

bwght2 = get_dataset("bwght2")
lbwght_multi1 = lm(@formula(lbwght ~ npvis + npvis^2), bwght2)
nobs(lbwght_multi1)
r2(lbwght_multi1)
- coef(lbwght_multi1)[2] / (2 * coef(lbwght_multi1)[3])
count(>(22), skipmissing(bwght2.npvis))

lbwght_multi2 = lm(@formula(lbwght ~ npvis + npvis^2 + mage + mage^2), bwght2)
nobs(lbwght_multi2)
r2(lbwght_multi2)
- coef(lbwght_multi2)[4] / (2 * coef(lbwght_multi2)[5])
count(>(30), skipmissing(bwght2.mage))

bwght_multi = lm(@formula(bwght ~ npvis + npvis^2 + mage + mage^2), bwght2)
nobs(bwght_multi)
r2(bwght_multi)

α1 = mean(exp, residuals(lbwght_multi2))
m = exp.(predict(lbwght_multi2))
# note that we had missing data, we could alternative dropmissing from data frame
y = bwght_multi.mf.data.bwght
α2 = (m' * y) / (m' * m)

cor(m, y) ^ 2
1 - sum((α1*m - y) .^ 2) / sum((y .- mean(y)) .^ 2)
1 - sum((α2*m - y) .^ 2) / sum((y .- mean(y)) .^ 2)

# Problem C11

apple = get_dataset("apple")

ecolbs_multi1 = lm(@formula(ecolbs ~ ecoprc + regprc), apple)
nobs(ecolbs_multi1)
r2(ecolbs_multi1)
adjr2(ecolbs_multi1)

describe(apple.ecolbs)
mean(apple.ecolbs .== 0)
describe(predict(ecolbs_multi1))

ecolbs_multi2 = lm(@formula(ecolbs ~ ecoprc + regprc + faminc + hhsize + educ + age), apple)

ftest(ecolbs_multi1.model, ecolbs_multi2.model)

ecolbs_ecoprc = lm(@formula(ecolbs ~ ecoprc), apple)
ecolbs_regprc = lm(@formula(ecolbs ~ regprc), apple)
cor(apple.ecoprc, apple.regprc)

ecolbs_multi3 = lm(@formula(ecolbs ~ ecoprc + regprc + faminc + reglbs), apple)
adjr2(ecolbs_multi3)
adjr2(lm(@formula(ecolbs ~ regprc + faminc + reglbs), apple))
adjr2(lm(@formula(ecolbs ~ ecoprc + faminc + reglbs), apple))
adjr2(lm(@formula(ecolbs ~ ecoprc + regprc + reglbs), apple))
adjr2(lm(@formula(ecolbs ~ ecoprc + regprc + faminc), apple))

# Problem C12

k401ksubs = get_dataset("k401ksubs")
k401ksubs_s = @rsubset(k401ksubs, :fsize == 1)
count(==(25), k401ksubs_s.age)

nettfa_multi1 = lm(@formula(nettfa ~ inc + age + age^2), k401ksubs_s)
- coef(nettfa_multi1)[3] / (2 * coef(nettfa_multi1)[4])

nettfa_multi2 = lm(@formula(nettfa ~ inc + age + (age-25)^2), k401ksubs_s)
nettfa_multi3 = lm(@formula(nettfa ~ inc + (age-25)^2), k401ksubs_s)
ftest(nettfa_multi3.model, nettfa_multi2.model)
ages = range(extrema(k401ksubs_s.age)...)
p = predict(nettfa_multi3, DataFrame(inc=30, age=ages))
plot(ages, p; xlabel="age", ylabel="predicted nettfa", label=false)

nettfa_multi4 = lm(@formula(nettfa ~ inc + inc^2 + (age-25)^2), k401ksubs_s)
ftest(nettfa_multi3.model, nettfa_multi4.model)

# Problem C13

meap00_01 = get_dataset("meap00_01")
math4_multi1 = lm(@formula(math4 ~ lexppp + lenroll + lunch), meap00_01)
p = predict(math4_multi1)
describe(p)
describe(meap00_01.math4)
u = residuals(math4_multi1)
meap00_01[argmax(u), :]
meap00_01[argmin(u), :]

math4_multi2 = lm(@formula(math4 ~ lexppp + lenroll + lunch + lexppp^2 + lenroll^2 + lunch^2), meap00_01)
ftest(math4_multi1.model, math4_multi2.model)

math4_multi3 = lm(@formula(math4 ~ lexppp + lenroll + lunch), mapcols(zscore, meap00_01))

# Problem C13

benefits = get_dataset("benefits")

lavgsal_bs = lm(@formula(lavgsal ~ bs), benefits)
r2(lavgsal_bs)

lm(@formula(identity(lavgsal+bs) ~ bs), benefits)

benefits.lbs = log.(benefits.bs)
describe(benefits, :detailed, cols=[:bs, :lbs])

lavgsal_lbs = lm(@formula(lavgsal ~ lbs), benefits)
r2(lavgsal_lbs)

lavgsal_multi1 = lm(@formula(lavgsal ~ bs + lenroll + lstaff + lunch), benefits)
lavgsal_multi2 = lm(@formula(lavgsal ~ bs + lenroll + lstaff + lunch + lunch^2), benefits)

x = - coef(lavgsal_multi2)[5] / (2 * coef(lavgsal_multi2)[6])
count(>(x), benefits.lunch)

mbenefits = mapcols(mean, benefits)
repeat!(mbenefits, 101)
mbenefits.lunch = 0:100
plot(0:100, predict(lavgsal_multi2, mbenefits);
     xlabel="lunch", ylabel="predicted lavgsal", label=false)
