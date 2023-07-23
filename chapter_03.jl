include("init_example.jl")

# Example 3.1 and 3.4

gpa1 = get_dataset("gpa1")
colGPA_hsGPA_ACT = lm(@formula(colGPA ~ hsGPA + ACT), gpa1)
nobs(colGPA_hsGPA_ACT)
r2(colGPA_hsGPA_ACT)
colGPA_ACT = lm(@formula(colGPA ~ ACT), gpa1)
nobs(colGPA_ACT)
r2(colGPA_ACT)

# Example 3.2 and 3.6 and problem C5

wage1 = get_dataset("wage1")
lwage_educ_exper_tenure = lm(@formula(lwage ~ educ + exper + tenure), wage1)
nobs(lwage_educ_exper_tenure)
r2(lwage_educ_exper_tenure)

lwage_educ = lm(@formula(lwage ~ educ), wage1)
nobs(lwage_educ)
r2(lwage_educ)

educ_exper_tenure = lm(@formula(educ ~ exper + tenure), wage1)
wage1.r1 = residuals(educ_exper_tenure)
lwage_r1 = lm(@formula(lwage ~ r1), wage1)

# Example 3.3

k401k = get_dataset("k401k")
prate_mrate_age = lm(@formula(prate ~ mrate + age), k401k)
nobs(prate_mrate_age)
r2(prate_mrate_age)

prate_mrate = lm(@formula(prate ~ mrate), k401k)
nobs(prate_mrate)
r2(prate_mrate)

cor(k401k.mrate, k401k.age)

# Example 3.5

crime1 = get_dataset("crime1")
narr86_pcnv_ptime86_qemp86 = lm(@formula(narr86 ~ pcnv + ptime86 + qemp86), crime1)
nobs(narr86_pcnv_ptime86_qemp86)
r2(narr86_pcnv_ptime86_qemp86)
narr86_pcnv_avgsen_ptime86_qemp86 = lm(@formula(narr86 ~ pcnv + avgsen + ptime86 + qemp86), crime1)
nobs(narr86_pcnv_avgsen_ptime86_qemp86)
r2(narr86_pcnv_avgsen_ptime86_qemp86)

# Example 3.7

jtrain98 = get_dataset("jtrain98")
earn98_train = lm(@formula(earn98 ~ train), jtrain98)
nobs(earn98_train)
r2(earn98_train)

earn98_train_earn96_educ_age_married = lm(@formula(earn98 ~ train + earn96 + educ + age + married), jtrain98)
nobs(earn98_train_earn96_educ_age_married)
r2(earn98_train_earn96_educ_age_married)

# Problem 1

gpa2 = get_dataset("gpa2")
colgpa_hsperc_sat = lm(@formula(colgpa ~ hsperc + sat), gpa2)
nobs(colgpa_hsperc_sat)
r2(colgpa_hsperc_sat)
predict(colgpa_hsperc_sat, DataFrame(hsperc=20, sat=1050))

# Problem 2 and C6

wage2 = get_dataset("wage2")
educ_sibs_meduc_feduc = lm(@formula(educ ~ sibs + meduc + feduc), wage2)
nobs(educ_sibs_meduc_feduc)
r2(educ_sibs_meduc_feduc)
predict(educ_sibs_meduc_feduc, DataFrame(sibs=0, meduc=[12, 16], feduc=[12, 16]))

IQ_educ = lm(@formula(IQ ~ educ), wage2)
δ₁ = coef(IQ_educ)[2]
lwage_educ = lm(@formula(lwage ~ educ), wage2)
β₁ = coef(lwage_educ)[2]
lwage_educ_IQ = lm(@formula(lwage ~ educ + IQ), wage2)
γ₁, γ₂ = coef(lwage_educ_IQ)[2:3]
γ₁ + γ₂*δ₁, β₁

# Problem 3

sleep75 = get_dataset("sleep75")
sleep_totwrk_educ_age = lm(@formula(sleep ~ totwrk + educ + age), sleep75)
nobs(sleep_totwrk_educ_age)
r2(sleep_totwrk_educ_age)

# Problem 4 and 16

lawsch85 = get_dataset("lawsch85")
lsalary_multi = lm(@formula(lsalary ~ LSAT + GPA + llibvol + lcost + rank), lawsch85)
nobs(lsalary_multi)
r2(lsalary_multi)

lsalary_rank_GPA = lm(@formula(lsalary ~ rank + GPA), lawsch85)
nobs(lsalary_rank_GPA)
r2(lsalary_rank_GPA)
lsalary_rank_GPA_age = lm(@formula(lsalary ~ rank + GPA + age), lawsch85)
nobs(lsalary_rank_GPA_age)
r2(lsalary_rank_GPA_age)

# Problem 9

hprice2 = get_dataset("hprice2")
lprice_lnox = lm(@formula(lprice ~ lnox), hprice2)
nobs(lprice_lnox)
r2(lprice_lnox)
lprice_lnox_rooms = lm(@formula(lprice ~ lnox + rooms), hprice2)
nobs(lprice_lnox_rooms)
r2(lprice_lnox_rooms)

# Problem 15

mlb1 = get_dataset("mlb1")
lsalary_years = lm(@formula(lsalary ~ years), mlb1)
nobs(lsalary_years)
r2(lsalary_years)
deviance(lsalary_years)
ser(lsalary_years)


lsalary_years_rbisyr = lm(@formula(lsalary ~ years + rbisyr), mlb1)
nobs(lsalary_years_rbisyr)
r2(lsalary_years_rbisyr)
deviance(lsalary_years_rbisyr)
ser(lsalary_years_rbisyr)
cor(mlb1.years, mlb1.rbisyr)
1 / (1 - cor(mlb1.years, mlb1.rbisyr)^2)

# Problem C1

bwght = get_dataset("bwght")
bwght_cigs_faminc = lm(@formula(bwght ~ cigs + faminc), bwght)
nobs(bwght_cigs_faminc)
r2(bwght_cigs_faminc)
bwght_cigs = lm(@formula(bwght ~ cigs), bwght)
nobs(bwght_cigs)
r2(bwght_cigs)

# Problem C2

hprice1 = get_dataset("hprice1")
price_sqrft_bdrms = lm(@formula(price ~ sqrft + bdrms), hprice1)
nobs(price_sqrft_bdrms)
r2(price_sqrft_bdrms)
predict(price_sqrft_bdrms, DataFrame(sqrft=2438, bdrms=4))

# Problem C3

ceosal2 = get_dataset("ceosal2")
lsalary_lsales_lmktval = lm(@formula(lsalary ~ lsales + lmktval), ceosal2)
nobs(lsalary_lsales_lmktval)
r2(lsalary_lsales_lmktval)

lsalary_lsales_lmktval_profits = lm(@formula(lsalary ~ lsales + lmktval + profits), ceosal2)
nobs(lsalary_lsales_lmktval_profits)
r2(lsalary_lsales_lmktval_profits)

lsalary_multi2 = lm(@formula(lsalary ~ lsales + lmktval + profits + ceoten), ceosal2)
nobs(lsalary_multi2)
r2(lsalary_multi2)
cor(ceosal2.lmktval, ceosal2.profits)

# Problem C4

attend = get_dataset("attend")
describe(attend)
atndrte_priGPA_ACT = lm(@formula(atndrte ~ priGPA + ACT), attend)
nobs(atndrte_priGPA_ACT)
r2(atndrte_priGPA_ACT)
predict(atndrte_priGPA_ACT, DataFrame(priGPA=[3.65, 3.1, 2.1], ACT=[20, 21, 26]))

# Problem C7

meap93 = get_dataset("meap93")
math10_lexpend_lnchprg = lm(@formula(math10 ~ lexpend + lnchprg), meap93)
nobs(math10_lexpend_lnchprg)
r2(math10_lexpend_lnchprg)

math10_lexpend= lm(@formula(math10 ~ lexpend), meap93)
nobs(math10_lexpend)
r2(math10_lexpend)
cor(meap93.lexpend, meap93.lnchprg)

# Problem C8

discrim = get_dataset("discrim")
describe(discrim, :mean, :std; cols=[:prpblck, :income])

psoda_prpblck_income = lm(@formula(psoda ~ prpblck + income), discrim)
nobs(psoda_prpblck_income)
r2(psoda_prpblck_income)

psoda_prpblck = lm(@formula(psoda ~ prpblck), discrim)
nobs(psoda_prpblck)
r2(psoda_prpblck)

lpsoda_prpblck_lincome = lm(@formula(lpsoda ~ prpblck + lincome), discrim)
nobs(lpsoda_prpblck_lincome)
r2(lpsoda_prpblck_lincome)

lpsoda_multi = lm(@formula(lpsoda ~ prpblck + lincome + prppov), discrim)
nobs(lpsoda_multi)
r2(lpsoda_multi)
discrim2 = select(discrim, :prppov, :lincome, :lpsoda, :prpblck)
pairwise(cor, eachcol(discrim2), skipmissing=:pairwise)
pairwise(cor, eachcol(discrim2), skipmissing=:listwise)

# Problem C9

charity = get_dataset("charity")

gift_multi = lm(@formula(gift ~ mailsyear + giftlast + propresp), charity)
nobs(gift_multi)
r2(gift_multi)

gift_mailsyear = lm(@formula(gift ~ mailsyear), charity)
nobs(gift_mailsyear)
r2(gift_mailsyear)

gift_multi2 = lm(@formula(gift ~ mailsyear + giftlast + propresp + avggift), charity)
nobs(gift_multi2)
r2(gift_multi2)

cor(charity.avggift, charity.giftlast)

# Problem C10

htv = get_dataset("htv")
extrema(htv.educ)
mean(htv.educ .== 12)
combine(htv, Cols(contains("educ")) .=> mean)

educ_motheduc_fatheduc = lm(@formula(educ ~ motheduc + fatheduc), htv)
nobs(educ_motheduc_fatheduc)
r2(educ_motheduc_fatheduc)

educ_multi1 = lm(@formula(educ ~ motheduc + fatheduc + abil), htv)
nobs(educ_multi1)
r2(educ_multi1)

educ_multi2 = lm(@formula(educ ~ motheduc + fatheduc + abil + abil^2), htv)
nobs(educ_multi2)
r2(educ_multi2)
β₃, β₄ = coef(educ_multi2)[4:5]
min_educ = -β₃ / (2 * β₄)
mean(htv.abil .< min_educ)
pred_htv = DataFrame(motheduc=12.18, fatheduc=12.45,
                     abil=minimum(htv.abil):0.01:maximum(htv.abil))
pred_htv.educ_hat = predict(educ_multi2, pred_htv)
plot(pred_htv.abil, pred_htv.educ_hat;
     legend=false, xlabel="abil", ylabel="predicted educ")
scatter!([min_educ], predict(educ_multi2,
                             DataFrame(motheduc=12.18, fatheduc=12.45, abil=min_educ)))

# Problem C11

meapsingle = get_dataset("meapsingle")

math4_pctsgle = lm(@formula(math4 ~ pctsgle), meapsingle)
nobs(math4_pctsgle)
r2(math4_pctsgle)

math4_multi = lm(@formula(math4 ~ pctsgle + lmedinc + free), meapsingle)
nobs(math4_multi)
r2(math4_multi)
cor(meapsingle.lmedinc, meapsingle.free)
1 / (1 - r2(lm(@formula(pctsgle ~ lmedinc + free), meapsingle)))
1 / (1 - r2(lm(@formula(lmedinc ~ pctsgle + free), meapsingle)))
1 / (1 - r2(lm(@formula(free ~ lmedinc + pctsgle), meapsingle)))
cor_mat = cor(Matrix(select(meapsingle, :pctsgle, :lmedinc, :free)))
diag(inv(cor_mat))

# Problem C12

econmath = get_dataset("econmath")
describe(econmath, :detailed; cols=[:score, :actmth, :acteng])

score_multi= lm(@formula(score ~ actmth + acteng + colgpa), econmath)
nobs(score_multi)
r2(score_multi)

# Problem C13

gpa1 = get_dataset("gpa1")

colGPA_PC = lm(@formula(colGPA ~ PC), gpa1)
nobs(colGPA_PC)
r2(colGPA_PC)

colGPA_multi1 = lm(@formula(colGPA ~ PC + hsGPA + ACT), gpa1)
nobs(colGPA_multi1)
r2(colGPA_multi1)

colGPA_multi2 = lm(@formula(colGPA ~ PC + hsGPA + ACT + fathcoll + mothcoll), gpa1)
nobs(colGPA_multi2)
r2(colGPA_multi2)

vif_df = DataFrame(variable=[:PC, :hsGPA, :ACT, :fathcoll, :mothcoll])
cor_gpa1 = cor(Matrix(select(gpa1, vif_df.variable)))
insertcols!(vif_df, :vif => diag(inv(cor_gpa1)))

function coef_vif(model)
    df = DataFrame(coeftable(model))
    mat = model.model.pp.X
    cor_mat = cor(mat)
    df.vif = diag(inv(ifelse.(isnan.(cor_mat), 0.0, cor_mat)))
    return df
end

coef_vif(colGPA_multi2)

