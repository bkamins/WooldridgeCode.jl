include("init_example.jl")

# Example 7.1 and 7.5 and 7.6

wage1 = get_dataset("wage1")
wage_multi = lm(@formula(wage ~ female + educ + exper + tenure), wage1)
wage_female = lm(@formula(wage ~ female), wage1)

lwage_multi = lm(@formula(lwage ~ female + educ + exper + exper^2 + tenure + tenure^2), wage1)

# note that source columns are integers
@rtransform!(wage1,
             :marrmale = (:female == 0) * :married,
             :marrfemale = :female * :married,
             :singmale = (:female == 0) * (:married == 0),
             :singfemale = :female * (:married == 0))
lm(@formula(lwage ~ marrmale + marrfemale + singfemale + educ + exper + exper^2 + tenure + tenure^2), wage1)
lm(@formula(lwage ~ marrmale + singmale + singfemale + educ + exper + exper^2 + tenure + tenure^2), wage1)

# Example 7.2 and Problem 5

gpa1 = get_dataset("gpa1")
colGPA_multi1 = lm(@formula(colGPA ~ PC + hsGPA + ACT), gpa1)
r2(colGPA_multi1)
colGPA_multi2 = lm(@formula(colGPA ~ PC), gpa1)
colGPA_multi3 = lm(@formula(colGPA ~ identity(1-PC) + hsGPA + ACT), gpa1)
r2(colGPA_multi3)

# Example 7.3

jtrain = get_dataset("jtrain")
hrsemp_multi = lm(@formula(hrsemp ~ grant + lsales + lemploy), jtrain)

# Example 7.4

hprice1 = get_dataset("hprice1")
lprice_multi = lm(@formula(lprice ~ llotsize + lsqrft + bdrms + colonial), hprice1)

# Example 7.8

lawsch85 = get_dataset("lawsch85")
@rtransform!(lawsch85,
             :top10 = :rank <= 10,
             :r11_25 = 11 <= :rank <= 25,
             :r26_40 = 26 <= :rank <= 40,
             :r41_60 = 41 <= :rank <= 60,
             :r61_100 = 61 <= :rank <= 100)
dropmissing!(lawsch85, [:rank, :LSAT, :GPA, :llibvol, :lcost])
lsalary_multi1 = lm(@formula(lsalary ~ top10 + r11_25 + r26_40 + r41_60 + r61_100 + LSAT + GPA + llibvol + lcost), lawsch85)
adjr2(lsalary_multi1)

lsalary_multi2 = lm(@formula(lsalary ~ rank + LSAT + GPA + llibvol + lcost), lawsch85)
adjr2(lsalary_multi2)


lsalary_multi3 = lm(@formula(lsalary ~ top10 + r11_25 + r26_40 + r41_60 + r61_100), lawsch85)
ftest(lsalary_multi3.model, lsalary_multi1.model)

lsalary_multi4 = lm(@formula(lsalary ~ rank), lawsch85)
ftest(lsalary_multi4.model, lsalary_multi2.model)

# Section 7-4a and Example 7.10

wage1 = get_dataset("wage1")
lwage_multi1 = lm(@formula(lwage ~ female*married + educ + exper + exper^2 + tenure + tenure^2), wage1)

lwage_multi2 = lm(@formula(lwage ~ female*educ + exper + exper^2 + tenure + tenure^2), wage1)

# Example 7.11 and Problem 12

mlb1 = get_dataset("mlb1")
dropmissing!(mlb1, [:years, :gamesyr, :bavg, :hrunsyr, :rbisyr, :runsyr, :fldperc, :allstar, :black, :hispan, :percblck, :perchisp])
lsalary_multi1 = lm(@formula(lsalary ~ years + gamesyr + bavg + hrunsyr + rbisyr + runsyr + fldperc + allstar + black + hispan + black&percblck + hispan&perchisp), mlb1)

lsalary_multi2 = lm(@formula(lsalary ~ years + gamesyr + bavg + hrunsyr + rbisyr + runsyr + fldperc + allstar), mlb1)
ftest(lsalary_multi2.model, lsalary_multi1.model)

mlb1.cpercblck = mlb1.percblck .- mean(mlb1.percblck)
mlb1.cperchisp = mlb1.perchisp .- mean(mlb1.perchisp)
lsalary_multi3 = lm(@formula(lsalary ~ years + gamesyr + bavg + hrunsyr + rbisyr + runsyr + fldperc + allstar + black + hispan + black&cpercblck + hispan&cperchisp), mlb1)

# Section 7-4c

gpa3 = get_dataset("gpa3")

cumgpa_multi1 = lm(@formula(cumgpa ~ female * (sat + hsperc + tothrs)),
                  @rsubset(gpa3, :term == 2))
nobs(cumgpa_multi1)
r2(cumgpa_multi1)
adjr2(cumgpa_multi1)

cumgpa_multi2 = lm(@formula(cumgpa ~ sat + hsperc + tothrs),
                  @rsubset(gpa3, :term == 2))
nobs(cumgpa_multi2)
r2(cumgpa_multi2)
adjr2(cumgpa_multi2)
ftest(cumgpa_multi2.model, cumgpa_multi1.model)

SSRₚ = deviance(cumgpa_multi2)
SSR₁ = deviance(lm(@formula(cumgpa ~ sat + hsperc + tothrs),
                   @rsubset(gpa3, :term == 2, :female == 1)))
SSR₂ = deviance(lm(@formula(cumgpa ~ sat + hsperc + tothrs),
                   @rsubset(gpa3, :term == 2, :female == 0)))

F = (SSRₚ - (SSR₁ + SSR₂)) / (SSR₁ + SSR₂) * (nobs(cumgpa_multi2) - 2 * 4) / 4
ccdf(FDist(4, nobs(cumgpa_multi2) - 2 * 4), F)

cumgpa_multi4 = lm(@formula(cumgpa ~ sat + hsperc + tothrs),
                  @rsubset(gpa3, :term == 2))
ftest(cumgpa_multi4.model, cumgpa_multi1.model)

cumgpa_multi3 = lm(@formula(cumgpa ~ female + sat + hsperc + tothrs),
                  @rsubset(gpa3, :term == 2))
ftest(cumgpa_multi3.model, cumgpa_multi1.model)


# Section 7.5 and Problem 7

mroz = get_dataset("mroz")
inlf_multi = lm(@formula(inlf ~ nwifeinc + educ + exper + exper^2 + age + kidslt6 + kidsge6), mroz)
nobs(inlf_multi)
r2(inlf_multi)

mroz.outlf = 1 .- mroz.inlf
outlf_multi = lm(@formula(outlf ~ nwifeinc + educ + exper + exper^2 + age + kidslt6 + kidsge6), mroz)
nobs(outlf_multi)
r2(outlf_multi)

# Example 7.12

crime1 = get_dataset("crime1")
crime1.arr86 = crime1.narr86 .> 0
arr86_multi1 = lm(@formula(arr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86), crime1)
nobs(arr86_multi1)
r2(arr86_multi1)

arr86_multi2 = lm(@formula(arr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + black + hispan), crime1)
nobs(arr86_multi2)
r2(arr86_multi2)

predict(arr86_multi2, DataFrame(pcnv=0, avgsen=0, tottime=0, ptime86=0, qemp86=4, black=1, hispan=0))

# Section 7-6

jtrain = get_dataset("jtrain")
lm(@formula(lscrap ~ grant + lsales + lemploy), @rsubset(jtrain, :year == 1988))

# Example 7.13

jtrain98 = get_dataset("jtrain98")
transform!(jtrain98, [:earn96, :educ, :age, :married] .=> (x -> x .- mean(x)) .=> n -> "c" * n)
earn98_multi1 = lm(@formula(earn98 ~ train + earn96 + educ + age + married +
                           train & (cearn96 + ceduc + cage + cmarried)), jtrain98)
nobs(earn98_multi1)
r2(earn98_multi1)

earn98_multi2 = lm(@formula(earn98 ~ train + earn96 + educ + age + married), jtrain98)
ftest(earn98_multi2.model, earn98_multi1.model)

# Section 7.7

fertil2 = get_dataset("fertil2")
lm(@formula(children ~ age + educ), fertil2)
lm(@formula(children ~ age + educ + electric), fertil2)

# Problem 1

sleep75 = get_dataset("sleep75")

sleep_multi1 = lm(@formula(sleep ~ totwrk + educ + age + age^2 + male), sleep75)
sleep_multi2 = lm(@formula(sleep ~ totwrk + educ + male), sleep75)
ftest(sleep_multi2.model, sleep_multi1.model)

# Problem 2

bwght = get_dataset("bwght")
lbwght_multi1 = lm(@formula(lbwght ~ cigs + lfaminc + parity + male + white), bwght)
lbwght_multi2 = lm(@formula(lbwght ~ cigs + lfaminc + parity + male + white + motheduc + fatheduc), bwght)
bwght_s = dropmissing(bwght, [:motheduc, :fatheduc])
lbwght_multi3 = lm(@formula(lbwght ~ cigs + lfaminc + parity + male + white), bwght_s)
ftest(lbwght_multi3.model, lbwght_multi2.model)

# Problem 3

gpa2 = get_dataset("gpa2")

sat_multi = lm(@formula(sat ~ hsize + hsize^2 + female*black), gpa2)
- coef(sat_multi)[2] / (2 * coef(sat_multi)[3])
sat_multi = lm(@formula(sat ~ hsize + hsize^2 + female + black + identity((1-female)*black)), gpa2)

# Problem 4

ceosal1 = get_dataset("ceosal1")
lsalary_multi = lm(@formula(lsalary ~ lsales + roe + finance + consprod + utility), ceosal1)
exp(coef(lsalary_multi)[6]) - 1

lm(@formula(lsalary ~ lsales + roe + finance + identity(finance+consprod) + utility), ceosal1)

# Problem 9

twoyear = get_dataset("twoyear")
lwage_multi = lm(@formula(lwage ~ female*totcoll), twoyear)
-coef(lwage_multi)[2] / coef(lwage_multi)[4]
combine(groupby(twoyear, :female), :totcoll => maximum)
unstack(twoyear, [], :female, :totcoll, combine=maximum)

# Problem 11

econmath = get_dataset("econmath")

score_multi1 = lm(@formula(score ~ colgpa), econmath)
score_multi2 = lm(@formula(score ~ male + colgpa), econmath)
score_multi3 = lm(@formula(score ~ male*colgpa), econmath)
ftest(score_multi1.model, score_multi3.model)

score_multi4 = lm(@formula(score ~ male + colgpa + identity(male*(colgpa - 2.81))), econmath)
describe(econmath, cols=:colgpa)

# Problem C1

gpa1 = get_dataset("gpa1")

colGPA_multi = lm(@formula(colGPA ~ PC + hsGPA + ACT + mothcoll + fathcoll), gpa1)
colGPA_multi_ref = lm(@formula(colGPA ~ PC + hsGPA + ACT), gpa1)
ftest(colGPA_multi_ref.model, colGPA_multi.model)

colGPA_multi_2 = lm(@formula(colGPA ~ PC + hsGPA + ACT + mothcoll + fathcoll + hsGPA^2), gpa1)

# Problem C2

wage2 = get_dataset("wage2")
lwage_multi1 = lm(@formula(lwage ~ educ + exper + tenure + married + black + south + urban), wage2)
lwage_multi2 = lm(@formula(lwage ~ educ + exper + tenure + married + black + south + urban + exper^2 + tenure^2), wage2)
ftest(lwage_multi1.model, lwage_multi2.model)

lwage_multi3 = lm(@formula(lwage ~ educ + educ&black + exper + tenure + married + black + south + urban), wage2)
lwage_multi4 = lm(@formula(lwage ~ educ + exper + tenure + married + black + south + urban +
                           educ&identity(married * (1-black)) + educ&identity((1-married) * black) +
                           educ&identity((1-married) * (1-black))), wage2)

# Problem C3

mlb1 = get_dataset("mlb1")

lsalary_multi1 = lm(@formula(lsalary ~ years + gamesyr + bavg + hrunsyr + rbisyr + runsyr + fldperc + allstar + frstbase + scndbase + thrdbase + shrtstop + catcher), mlb1)
lsalary_multi2 = lm(@formula(lsalary ~ years + gamesyr + bavg + hrunsyr + rbisyr + runsyr + fldperc + allstar), mlb1)
ftest(lsalary_multi2.model, lsalary_multi1.model)

# Problem C4

gpa2 = get_dataset("gpa2")

colgpa_multi1 = lm(@formula(colgpa ~ hsize + hsize^2 + hsperc + sat + female + athlete), gpa2)
colgpa_multi2 = lm(@formula(colgpa ~ hsize + hsize^2 + hsperc + female + athlete), gpa2)
cor(gpa2.athlete, gpa2.sat)

colgpa_multi3 = lm(@formula(colgpa ~ hsize + hsize^2 + hsperc + sat + athlete*female), gpa2)
colgpa_multi4 = lm(@formula(colgpa ~ hsize + hsize^2 + hsperc + sat*female + athlete), gpa2)

# Problem C5

ceosal1 = get_dataset("ceosal1")
lsalary_multi = lm(@formula(lsalary ~ lsales + roe + identity(ros<0)), ceosal1)

# Problem C6

sleep75 = get_dataset("sleep75")

sleep_men = lm(@formula(sleep ~ totwrk + educ + age + age^2 + yngkid), @rsubset(sleep75, :male == 1))
sleep_women = lm(@formula(sleep ~ totwrk + educ + age + age^2 + yngkid), @rsubset(sleep75, :male == 0))
sleep_multi1 = lm(@formula(sleep ~ totwrk + educ + age + age^2 + yngkid), sleep75)

SSRₚ = deviance(sleep_multi1)
SSR₁ = deviance(sleep_men)
SSR₂ = deviance(sleep_women)

F = (SSRₚ - (SSR₁ + SSR₂)) / (SSR₁ + SSR₂) * (nobs(sleep_multi) - 2 * 6) / 6
ccdf(FDist(6, nobs(sleep_multi) - 2 * 6), F)

sleep_multi2 = lm(@formula(sleep ~ male * (totwrk + educ + age + age^2 + yngkid)), sleep75)
ftest(sleep_multi1.model, sleep_multi2.model)

sleep_multi3 = lm(@formula(sleep ~ male + totwrk + educ + age + age^2 + yngkid), sleep75)
ftest(sleep_multi3.model, sleep_multi2.model)

# Problem C7

wage1 = get_dataset("wage1")
lwage_multi1 = lm(@formula(lwage ~ female*educ + exper + exper^2 + tenure + tenure^2), wage1)
p = predict(lwage_multi1, DataFrame(female=[0, 1], educ=12.5, exper=0, tenure=0))
diff(p)

p = predict(lwage_multi1, DataFrame(female=[0, 1], educ=0, exper=0, tenure=0))
diff(p)

lwage_multi2 = lm(@formula(lwage ~ female + educ + identity(female*(educ-12.5)) + exper + exper^2 + tenure + tenure^2), wage1)

# Problem C8

loanapp = get_dataset("loanapp")
lm(@formula(approve ~ white), loanapp)
lm(@formula(approve ~ white + hrat + obrat + loanprc + unem + male + married + dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr), loanapp)
lm(@formula(approve ~ white + hrat + obrat + loanprc + unem + male + married + dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr + white&obrat), loanapp)
lm(@formula(approve ~ white + hrat + obrat + loanprc + unem + male + married + dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr + identity(white*(obrat-32))), loanapp)

# Problem C9 and C11

k401ksubs = get_dataset("k401ksubs")
mean(k401ksubs.e401k)
e401k_multi = lm(@formula(e401k ~ inc + inc^2 + age + age^2 + male), k401ksubs)
p = describe(predict(e401k_multi))
k401ksubs.pe401k = predict(e401k_multi) .>= 0.5
combine(groupby(k401ksubs, :e401k), nrow, :pe401k => mean)
mean(k401ksubs.e401k .== k401ksubs.pe401k)

e401k_multi2 = lm(@formula(e401k ~ inc + inc^2 + age + age^2 + male + pira), k401ksubs)

describe(k401ksubs, :detailed; cols=:nettfa)
lm(@formula(nettfa ~ e401k), k401ksubs)
lm(@formula(nettfa ~ e401k + age + age^2 + inc + inc^2), k401ksubs)
lm(@formula(nettfa ~ e401k + age + age^2 + inc + inc^2 +
            identity(e401k*(age-41)) + identity(e401k*(age-41)^2)), k401ksubs)
k401ksubs.fsizes = string.("f", min.(k401ksubs.fsize, 5))

nettfa_multi_r = lm(@formula(nettfa ~ e401k + age + age^2 + inc + inc^2 + fsizes), k401ksubs)
nettfa_multi_f = [lm(@formula(nettfa ~ e401k + age + age^2 + inc + inc^2),
                     @rsubset(k401ksubs, :fsizes == "f$s")) for s in 1:5]

SSR_r = deviance(nettfa_multi_r)
SSR_f = deviance.(nettfa_multi_f)

F = (SSR_r - sum(SSR_f)) / sum(SSR_f) * (nobs(nettfa_multi_r) - 30) / 20
ccdf(FDist(20, nobs(nettfa_multi_r) - 30), F)

# Problem C10

nbasal = get_dataset("nbasal")

points_multi1 = lm(@formula(points ~ exper + exper^2 + guard + forward), nbasal)
points_multi2 = lm(@formula(points ~ exper + exper^2 + guard + forward + marr), nbasal)
points_multi3 = lm(@formula(points ~ marr*(exper + exper^2) + guard + forward), nbasal)
ftest(points_multi1.model, points_multi3.model)

assists_multi = lm(@formula(assists ~ exper + exper^2 + guard + forward + marr), nbasal)

# Problem C12

beauty = get_dataset("beauty")
combine(groupby(beauty, :female), nrow, [:belavg, :abvavg] .=> mean)
combine(beauty, [:belavg, :abvavg] .=> sum)

lm(@formula(abvavg ~ female), beauty)
lm(@formula(lwage ~ abvavg + belavg), @rsubset(beauty, :female == 0))
lm(@formula(lwage ~ abvavg + belavg), @rsubset(beauty, :female == 1))

lm(@formula(lwage ~ abvavg + belavg + educ + exper + exper^2 + union + goodhlth + black + married + south + bigcity + smllcity + service),
   @rsubset(beauty, :female == 0))
lm(@formula(lwage ~ abvavg + belavg + educ + exper + exper^2 + union + goodhlth + black + married + south + bigcity + smllcity + service),
   @rsubset(beauty, :female == 1))

lwage_multi = lm(@formula(lwage ~ female + abvavg + belavg + educ + exper + exper^2 +
                          union + goodhlth + black + married + south + bigcity + smllcity + service), beauty)
lwage_multi_ref = lm(@formula(lwage ~ female * (abvavg + belavg + educ + exper + exper^2 +
                              union + goodhlth + black + married + south + bigcity + smllcity + service)), beauty)
ftest(lwage_multi.model, lwage_multi_ref.model)

# Problem C13

apple = get_dataset("apple")
apple.ecobuy = apple.ecolbs .> 0
mean(apple.ecobuy)

ecobuy_multi = lm(@formula(ecobuy ~ ecoprc + regprc + faminc + hhsize + educ + age), apple)
r2(ecobuy_multi)

ecobuy_res = lm(@formula(ecobuy ~ ecoprc + regprc), apple)
ftest(ecobuy_res.model, ecobuy_multi.model)

ecobuy_multi2 = lm(@formula(ecobuy ~ ecoprc + regprc + log(faminc) + hhsize + educ + age), apple)
r2(ecobuy_multi2)
p = predict(ecobuy_multi2)
extrema(p)
count(>(1), p)
nobs(ecobuy_multi2)
p[findall(>(1), p)]

t = 0.5
apple.pecobuy = p .> t
proptable(apple, :ecobuy, :pecobuy, margins=1)

# Problem C14

charity = get_dataset("charity")
respond_multi1 = lm(@formula(respond ~ resplast + avggift), charity)
respond_multi2 = lm(@formula(respond ~ resplast + avggift + propresp), charity)
respond_multi3 = lm(@formula(respond ~ resplast + avggift + propresp + mailsyear), charity)

# Problem C15

fertil2 = get_dataset("fertil2")
describe(fertil2, :detailed, cols=:children)
mean(skipmissing(fertil2.electric))
combine(groupby(fertil2, :electric), :children => mean)
lm(@formula(children ~ electric), fertil2)
lm(@formula(children ~ electric + age^2 + urban + spirit + protest + catholic), fertil2)
lm(@formula(children ~ electric + electric&educ + age^2 + urban + spirit + protest + catholic), fertil2)
lm(@formula(children ~ electric + identity(electric*(educ-7)) + age^2 + urban + spirit + protest + catholic), fertil2)

# Problem C16

catholic = get_dataset("catholic")
mean(catholic.cathhs)
mean(catholic.math12)
math12_cathhs = lm(@formula(math12 ~ cathhs), catholic)
nobs(math12_cathhs)
math12_multi1 = lm(@formula(math12 ~ cathhs + lfaminc + motheduc + fatheduc), catholic)
nobs(math12_multi1)
math12_multi2 = lm(@formula(math12 ~ cathhs * (lfaminc + motheduc + fatheduc)), catholic)
ftest(math12_multi1.model, math12_multi2.model)
df_pred = mapcols(mean, catholic)
repeat!(df_pred, 2)
df_pred.cathhs = [0, 1]
p = predict(math12_multi2, df_pred)
diff(p)

# Problem C17

jtrain98 = get_dataset("jtrain98")
combine(groupby(jtrain98, :train), [:unem96, :unem98] .=> mean)
lm(@formula(unem98 ~ train), jtrain98)
unem98_multi1 = lm(@formula(unem98 ~ train + earn96 + educ + age + married), jtrain98)
transform!(jtrain98, [:earn96, :educ, :age, :married] .=> (x -> x .- mean(x)) .=> n -> "c" * n)
unem98_multi2 = lm(@formula(unem98 ~ train * (cearn96 + ceduc + cage + cmarried)), jtrain98)
ftest(unem98_multi1.model, unem98_multi2.model)

unem98_multi3 = lm(@formula(unem98 ~ earn96 + educ + age + married),
                   @rsubset(jtrain98, :train == 0))
unem98_multi4 = lm(@formula(unem98 ~ earn96 + educ + age + married),
                   @rsubset(jtrain98, :train == 1))
p0 = mean(predict(unem98_multi3, jtrain98))
p1 = mean(predict(unem98_multi4, jtrain98))
p1 - p0
