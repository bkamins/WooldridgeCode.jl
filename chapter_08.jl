include("init_example.jl")

# Example 8.1

wage1 = get_dataset("wage1")
@rtransform!(wage1,
             :marrmale = (:female == 0) * :married,
             :marrfemale = :female * :married,
             :singmale = (:female == 0) * (:married == 0),
             :singfemale = :female * (:married == 0))
lwage_multi = lm(@formula(lwage ~ marrmale + marrfemale + singfemale + educ + exper + exper^2 + tenure + tenure^2), wage1)
lm_ttest(lwage_multi)
lm_ttest(lwage_multi, :HC0)
lm_ttest(lwage_multi, :HC1)
lm_ttest(lwage_multi, :HC2)
lm_ttest(lwage_multi, :HC3)

# Example 8.2 and Problem 4

gpa3 = get_dataset("gpa3")
cumgpa_multi = lm(@formula(cumgpa ~ sat + hsperc + tothrs + female + black + white),
                  @rsubset(gpa3, :term == 2))
nobs(cumgpa_multi)
r2(cumgpa_multi)
adjr2(cumgpa_multi)
lm_ttest(cumgpa_multi)
lm_ttest(cumgpa_multi, :HC0)
lm_ttest(cumgpa_multi, :HC1)
lm_ttest(cumgpa_multi, :HC2)
lm_ttest(cumgpa_multi, :HC3)

cumgpa_multi2 = lm(@formula(cumgpa ~ sat + hsperc + tothrs + female),
                  @rsubset(gpa3, :term == 2))
ftest(cumgpa_multi2.model, cumgpa_multi.model)

lm_waldtest(cumgpa_multi,
            [0.0 0.0 0.0 0.0 0.0 1.0 0.0
             0.0 0.0 0.0 0.0 0.0 0.0 1.0],
            [0.0, 0.0])
lm_waldtest(cumgpa_multi,
            [0.0 0.0 0.0 0.0 0.0 1.0 0.0
             0.0 0.0 0.0 0.0 0.0 0.0 1.0],
            [0.0, 0.0], :HC0)
lm_waldtest(cumgpa_multi,
            [0.0 0.0 0.0 0.0 0.0 1.0 0.0
             0.0 0.0 0.0 0.0 0.0 0.0 1.0],
            [0.0, 0.0], :HC1)
lm_waldtest(cumgpa_multi,
            [0.0 0.0 0.0 0.0 0.0 1.0 0.0
             0.0 0.0 0.0 0.0 0.0 0.0 1.0],
            [0.0, 0.0], :HC2)
lm_waldtest(cumgpa_multi,
            [0.0 0.0 0.0 0.0 0.0 1.0 0.0
             0.0 0.0 0.0 0.0 0.0 0.0 1.0],
            [0.0, 0.0], :HC3)

cumgpa_multi2 = lm(@formula(trmgpa ~ crsgpa + cumgpa + tothrs + sat + hsperc + female + season),
                   @rsubset(gpa3, :frstsem == 0, :spring == 0))
lm_ttest(cumgpa_multi2, :HC0)

cumgpa_multi3 = lm(@formula(trmgpa-crsgpa ~ crsgpa + cumgpa + tothrs + sat + hsperc + female + season),
                   @rsubset(gpa3, :frstsem == 0, :spring == 0))
lm_ttest(cumgpa_multi3, :HC0)

# Example 8.3

crime1 = get_dataset("crime1")
narr86_multi_full = lm(@formula(narr86 ~ pcnv + avgsen + avgsen^2 + ptime86 + qemp86 + inc86 + black + hispan), crime1)
lm_ttest(narr86_multi_full, :HC0)

narr86_ref = lm(@formula(narr86 ~ pcnv + ptime86 + qemp86 + inc86 + black + hispan), crime1)
crime1.u = residuals(narr86_ref)
u_multi = lm(@formula(u ~ pcnv + avgsen + avgsen^2 + ptime86 + qemp86 + inc86 + black + hispan), crime1)
LM = nobs(u_multi) * r2(u_multi)
ccdf(Chisq(2), LM)

avgsen_multi = lm(@formula(avgsen ~ pcnv + ptime86 + qemp86 + inc86 + black + hispan), crime1)
avgsen2_multi = lm(@formula(avgsen^2 ~ pcnv + ptime86 + qemp86 + inc86 + black + hispan), crime1)

crime1.r1 = residuals(avgsen_multi)
crime1.r2 = residuals(avgsen2_multi)
LMR = nobs(narr86_multi_full) - deviance(lm(@formula(1 ~ 0 + identity(u*r1) + identity(u*r2)), crime1))
ccdf(Chisq(2), LMR)

# Example 8.4 and 8.5 and Problem C2 and C3

hprice1 = get_dataset("hprice1")
price_multi = lm(@formula(price ~ lotsize + sqrft + bdrms), hprice1)
lm_ttest(price_multi, :HC0)
hprice1.u = residuals(price_multi)
u2_multi = lm(@formula(u^2 ~ lotsize + sqrft + bdrms), hprice1)
r2(u2_multi)
nobs(u2_multi)
ftest(u2_multi.model)
LM = r2(u2_multi) * nobs(u2_multi)
ccdf(Chisq(3), LM)
lm_ttest(price_multi, :HC3)

lprice_multi = lm(@formula(lprice ~ llotsize + lsqrft + bdrms), hprice1)
lm_ttest(lprice_multi, :HC0)
hprice1.ul = residuals(lprice_multi)
ul2_multi = lm(@formula(ul^2 ~ llotsize + lsqrft + bdrms), hprice1)
r2(ul2_multi)
nobs(ul2_multi)
ftest(ul2_multi.model)
LM = r2(ul2_multi) * nobs(ul2_multi)
ccdf(Chisq(3), LM)
hprice1.lprice_hat = predict(lprice_multi)
ul2_multi2 = lm(@formula(ul^2 ~ lprice_hat + lprice_hat^2), hprice1)
ftest(ul2_multi2.model)
LM = r2(ul2_multi2) * nobs(ul2_multi2)
ccdf(Chisq(2), LM)

ul2_multi2 = lm(@formula(ul^2 ~ llotsize + lsqrft + bdrms +
                                llotsize^2 + lsqrft^2 + bdrms^2 +
                                llotsize*lsqrft + llotsize*bdrms + lsqrft*bdrms), hprice1)
ftest(ul2_multi2.model)


# Example 8.6 and Table 8.2 and Problem C11

k401ksubs = get_dataset("k401ksubs")
k401ksubs_s = @rsubset(k401ksubs, :fsize == 1)
nettfa_ols1 = lm(@formula(nettfa ~ inc), k401ksubs_s)

k401ksubs_s.u1 = residuals(nettfa_ols1)
u1_inc = lm(@formula(u1^2 ~ inc), k401ksubs_s)
ftest(u1_inc.model)

lm_ttest(nettfa_ols1, :HC0)

nettfa_wls1 = lm(@formula(nettfa ~ inc), k401ksubs_s; wts=awts(k401ksubs_s.inc))
nobs(nettfa_wls1)
r2(nettfa_wls1)

nettfa_ols2 = lm(@formula(nettfa ~ inc + identity((age-25)^2) + male + e401k), k401ksubs_s)

k401ksubs_s.u2 = residuals(nettfa_ols2)
u2_inc = lm(@formula(u2^2 ~ inc + identity((age-25)^2) + male + e401k), k401ksubs_s)
ftest(u2_inc.model)

lm_ttest(nettfa_ols2, :HC0)

nettfa_wls2 = lm(@formula(nettfa ~ inc + identity((age-25)^2) + male + e401k), k401ksubs_s; wts=awts(k401ksubs_s.inc))
nobs(nettfa_wls2)
r2(nettfa_wls2)
ftest(nettfa_wls1.model, nettfa_wls2.model)

lm_ttest(nettfa_wls2)
lm_ttest(nettfa_wls2, :HC0)

nettfa_ols3 = lm(@formula(nettfa ~ inc*e401k + identity((age-25)^2) + male), k401ksubs_s)
lm_ttest(nettfa_ols3, :HC0)
nettfa_wls3 = lm(@formula(nettfa ~ inc*e401k + identity((age-25)^2) + male), k401ksubs_s;
                 wts=awts(k401ksubs_s.inc))
lm_ttest(nettfa_wls3, :HC0)
nettfa_wls4 = lm(@formula(nettfa ~ inc + e401k + identity((inc-30)*e401k)+ identity((age-25)^2) + male), k401ksubs_s;
                 wts=awts(k401ksubs_s.inc))
lm_ttest(nettfa_wls4, :HC0)

# Example 8.7 and Problem 5 and C9

smoke = get_dataset("smoke")
cigs_multi1 = lm(@formula(cigs ~ lincome + lcigpric + educ + age + age^2 + restaurn), smoke)
smoke.u1 = residuals(cigs_multi1)
u12_multi = lm(@formula(u1^2 ~ lincome + lcigpric + educ + age + age^2 + restaurn), smoke)
r2(u12_multi)
ftest(u12_multi.model)
lu12_multi = lm(@formula(log(u1^2) ~ lincome + lcigpric + educ + age + age^2 + restaurn), smoke)
smoke.h1 = exp.(predict(lu12_multi))

cigs_wls1 = lm(@formula(cigs ~ lincome + lcigpric + educ + age + age^2 + restaurn), smoke; wts=awts(smoke.h1))
smoke.pred = predict(cigs_wls1)
smoke.u2 = residuals(cigs_wls1)
lm_ttest(cigs_wls1, :HC0)

ftest((lm(@formula((u2^2/h1) ~ pred/sqrt(h1) + pred^2/h1), smoke)).model)

smokes_multi = lm(@formula(cigs > 0 ~ lcigpric + lincome + educ + age + age^2 + restaurn + white), smoke)
lm_ttest(smokes_multi, :HC0)

-coef(smokes_multi)[5] / (2 * coef(smokes_multi)[6])

predict(smokes_multi)[206]

u22_multi = lm(@formula(u2^2/sqrt(h1) ~ pred/sqrt(h1) + pred^2/h1), smoke)
r2(u22_multi)
ftest(u22_multi.model)

# Example 8.8

mroz = get_dataset("mroz")
inlf_multi = lm(@formula(inlf ~ nwifeinc + educ + exper + exper^2 + age + kidslt6 + kidsge6), mroz)
nobs(inlf_multi)
r2(inlf_multi)
lm_ttest(inlf_multi)
lm_ttest(inlf_multi, :HC0)

# Example 8.9

gpa1 = get_dataset("gpa1")
PC_ols = lm(@formula(PC ~ hsGPA + ACT + max(mothcoll, fathcoll)), gpa1)
lm_ttest(PC_ols, :HC0)
pred = predict(PC_ols)
extrema(pred)
gpa1.wts = pred .* (1.0 .- pred)
PC_wls = lm(@formula(PC ~ hsGPA + ACT + max(mothcoll, fathcoll)), gpa1; wts=awts(gpa1.wts))
lm_ttest(PC_wls, :HC0)

# Problem 8

econmath = get_dataset("econmath")

score_multi1 = lm(@formula(score ~ colgpa + act), @rsubset(econmath, :male == 1))
lm_ttest(score_multi1, :HC0)
score_multi2 = lm(@formula(score ~ colgpa + act), @rsubset(econmath, :male == 0))
lm_ttest(score_multi2, :HC0)
score_multi3 = lm(@formula(score ~ male + colgpa + act), econmath)
lm_ttest(score_multi3, :HC0)
score_multi4 = lm(@formula(score ~ male * (colgpa + act)), econmath)
lm_ttest(score_multi4, :HC0)

chow1 = (deviance(score_multi3) / (deviance(score_multi1) + deviance(score_multi1)) - 1.0) * (nobs(score_multi1) - 4) / 8 
ccdf(FDist(4, nobs(score_multi1) - 4), chow1)

ftest(score_multi3.model, score_multi4.model)
lm_waldtest(score_multi4, [0.0 0.0 0.0 0.0 1.0 0.0
                           0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0])
lm_waldtest(score_multi4, [0.0 0.0 0.0 0.0 1.0 0.0
                           0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0], :HC0)
lm_waldtest(score_multi4, [0.0 0.0 0.0 0.0 1.0 0.0
                           0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0], :HC1)
lm_waldtest(score_multi4, [0.0 0.0 0.0 0.0 1.0 0.0
                           0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0], :HC2)
lm_waldtest(score_multi4, [0.0 0.0 0.0 0.0 1.0 0.0
                           0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0], :HC3)

# Problem C1

sleep75 = get_dataset("sleep75")
sleep_multi = lm(@formula(sleep ~ totwrk + educ + age + age^2 + yngkid + male), sleep75)
sleep75.u = residuals(sleep_multi)
u_male = lm(@formula(u^2 ~ male), sleep75)

# Problem C4

vote1 = get_dataset("vote1")
voteA_multi = lm(@formula(voteA ~ prtystrA + democA + lexpendA + lexpendB), vote1)
vote1.u = residuals(voteA_multi)
u_multi = lm(@formula(u ~ prtystrA + democA + lexpendA + lexpendB), vote1)
r2(u_multi)
u2_multi = lm(@formula(u^2 ~ prtystrA + democA + lexpendA + lexpendB), vote1)
r2(u2_multi)
ftest(u2_multi.model)
vote1.pred = predict(voteA_multi)
u2_multi2 = lm(@formula(u^2 ~ pred + pred^2), vote1)
r2(u2_multi2)
ftest(u2_multi2.model)

# Problem C5

pntsprd = get_dataset("pntsprd")
lm(@formula(sprdcvr-0.5 ~ 1), pntsprd)
freqtable(pntsprd, :neutral)

sprdcvr_ols = lm(@formula(sprdcvr ~ favhome + neutral + fav25 + und25), pntsprd)
lm_ttest(sprdcvr_ols, :HC0)
ftest(sprdcvr_ols.model)

# Problem C6

crime1 = get_dataset("crime1")
crime1.arr86 = crime1.narr86 .> 0
arr86_ols = lm(@formula(arr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86), crime1)
pred = predict(arr86_ols)
describe(pred)
wts = pred .* (1.0 .- pred)
arr86_wls = lm(@formula(arr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86), crime1;
               wts=awts(wts))
lm_waldtest(arr86_wls, [0.0 0.0 1.0 0.0 0.0 0.0
                        0.0 0.0 0.0 1.0 0.0 0.0], [0.0, 0.0])
lm_waldtest(arr86_wls, [0.0 0.0 1.0 0.0 0.0 0.0
                        0.0 0.0 0.0 1.0 0.0 0.0], [0.0, 0.0], :HC0)

# Problem C7

loanapp = get_dataset("loanapp")
approve_ols = lm(@formula(approve ~ white + hrat + obrat + loanprc + unem + male + married + dep + sch + cosign + white&obrat), loanapp)
lm_ttest(approve_ols, :HC0)
describe(predict(approve_ols))

# Problem C8

gpa1 = get_dataset("gpa1")
colGPA_ols = lm(@formula(colGPA ~ hsGPA + ACT + skipped + PC), gpa1)
gpa1.u = residuals(colGPA_ols)
gpa1.pred = predict(colGPA_ols)
u2_multi = lm(@formula(u^2 ~ pred + pred^2), gpa1)
ftest(u2_multi.model)
gpa1.h = predict(u2_multi)
describe(gpa1.h)
colGPA_wls = lm(@formula(colGPA ~ hsGPA + ACT + skipped + PC), gpa1; wts=awts(gpa1.h))
lm_ttest(colGPA_wls, :HC0)

# Problem C10

k401ksubs = get_dataset("k401ksubs")
e401k_ols = lm(@formula(e401k ~ inc + inc^2 + age + age^2 + male), k401ksubs)
lm_ttest(e401k_ols, :HC0)
k401ksubs.u = residuals(e401k_ols)
k401ksubs.pred = predict(e401k_ols)
u2_multi = lm(@formula(u^2 ~ pred + pred^2), k401ksubs)
ftest(u2_multi.model)
describe(k401ksubs.pred)
wts = k401ksubs.pred .* (1.0 .- k401ksubs.pred)
e401k_wls = lm(@formula(e401k ~ inc + inc^2 + age + age^2 + male), k401ksubs;
               wts=awts(wts))

# Problem C12

meap00_01 = get_dataset("meap00_01")
math4_ols = lm(@formula(math4 ~ lunch + lenroll + lexppp), meap00_01)
lm_ttest(math4_ols, :HC0)
meap00_01.u = residuals(math4_ols)
meap00_01.pred = predict(math4_ols)
ftest(lm(@formula(u^2 ~ pred + pred^2), meap00_01).model)
meap00_01.g = predict(lm(@formula(log(u^2) ~ pred + pred^2), meap00_01))
math4_wls = lm(@formula(math4 ~ lunch + lenroll + lexppp), meap00_01;
               wts=awts(exp.(meap00_01.g)))
lm_ttest(math4_wls, :HC0)

# Problem C13

fertil2 = get_dataset("fertil2")
children_ols = lm(@formula(children ~ age + age^2 + educ + electric + urban), fertil2)
lm_ttest(children_ols, :HC0)
children_ols2 = lm(@formula(children ~ age + age^2 + educ + electric + urban + spirit + protest + catholic), fertil2)
ftest(children_ols.model, children_ols2.model)
lm_waldtest(children_ols2, [0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
                            0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0
                            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0, 0.0])
lm_waldtest(children_ols2, [0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
                            0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0
                            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0], [0.0, 0.0, 0.0], :HC0)
u = residuals(children_ols2)
pred = predict(children_ols2)
u2_multi = lm(@formula(u^2 ~ pred + pred^2), (; u, pred))
ftest(u2_multi.model)

# Problem C14

beauty = get_dataset("beauty")
lwage_ols = lm(@formula(lwage ~ belavg + abvavg + female + educ + exper + exper^2), beauty)
lm_ttest(lwage_ols)
lwage_ols2 = lm(@formula(lwage ~ female * (belavg + abvavg + educ + exper + exper^2)), beauty)
ftest(lwage_ols.model, lwage_ols2.model)
lm_waldtest(lwage_ols2, [0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0],
                        zeros(5))
lm_waldtest(lwage_ols2, [0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0],
                        zeros(5), :HC0)

lm_waldtest(lwage_ols2, [0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0
                         0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0],
                        zeros(2), :HC0)

