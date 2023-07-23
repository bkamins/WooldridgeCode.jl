include("init_example.jl")

# Example 5.2

bwght = get_dataset("bwght")
lbwght_multi1 = lm(@formula(lbwght ~ cigs + lfaminc), bwght)
nobs(lbwght_multi1)
lbwght_multi2 = lm(@formula(lbwght ~ cigs + lfaminc), bwght[1:end÷2, :])
nobs(lbwght_multi2)

# Example 5.3

crime1 = get_dataset("crime1")
narr86_ref = lm(@formula(narr86 ~ pcnv + ptime86 + qemp86), crime1)
crime1.u = residuals(narr86_ref)
u_multi = lm(@formula(u ~ pcnv + avgsen + tottime + ptime86 + qemp86), crime1)
nobs(u_multi) * r2(u_multi)
quantile(Chisq(2), 0.9)

narr86_multi = lm(@formula(narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86), crime1)
ftest(narr86_ref.model, narr86_multi.model)

# Problem 3

smoke = get_dataset("smoke")
combine(groupby(smoke, :cigs, sort=true), proprow)
histogram(smoke.cigs; bins=80, label=false, xlabel="cigs")

# Problem 5 and Problem C6

econmath = get_dataset("econmath")
extrema(econmath.score)
histogram(econmath.score; bins=range(extrema(econmath.score)..., length=30), label=false,
          xlabel="course score (in percentage form)", ylabel="proportion in cell",
          normalize=:pdf)
score_m = mean(econmath.score)
score_s = std(econmath.score)
norm = Normal(score_m, score_s)
plot!(x -> pdf(norm, x), label="normal", linewidth=3)
ccdf(norm, 100)

score_multi = lm(@formula(score ~ colgpa + actmth + acteng), econmath)
u = residuals(score_multi)
p = predict(score_multi)
histogram(u; normalize=:pdf, label="residuals")
plot!(x -> pdf(Normal(mean(u), std(u)), x);
      label="normal", linewidth=3)
scatter(p, u, label="u")

# Problem C1

wage1 = get_dataset("wage1")
wage_multi1 = lm(@formula(wage ~ educ + exper + tenure), wage1)
u1 = residuals(wage_multi1)

wage_multi2 = lm(@formula(lwage ~ educ + exper + tenure), wage1)
u2 = residuals(wage_multi2)
plot(histogram(u1; label=false, title="residuals wage"),
     histogram(u2; label=false, title="residuals lwage"))

# Problem C2

gpa2 = get_dataset("gpa2")
colgpa_multi1 = lm(@formula(colgpa ~ hsperc + sat), gpa2)
nobs(colgpa_multi1)
r2(colgpa_multi1)

colgpa_multi2 = lm(@formula(colgpa ~ hsperc + sat), gpa2[1:2070, :])
nobs(colgpa_multi2)
r2(colgpa_multi2)
DataFrame(coeftable(colgpa_multi1))."Std. Error" * sqrt(nobs(colgpa_multi1))
DataFrame(coeftable(colgpa_multi2))."Std. Error" * sqrt(nobs(colgpa_multi2))

# Problem C3

bwght = get_dataset("bwght")
bwght2 = select(bwght, :bwght, :cigs, :parity, :faminc, :motheduc, :fatheduc)
dropmissing!(bwght2)
bwght_multi = lm(@formula(bwght ~ cigs + parity + faminc), bwght2)

bwght2.u = residuals(bwght_multi)
u_multi = lm(@formula(u ~ cigs + parity + faminc + motheduc + fatheduc), bwght2)
nobs(u_multi) * r2(u_multi)
quantile(Chisq(2), 0.9)

# Problem C4

skew(x) = mean(v -> v^3, zscore(x))

k401ksubs = get_dataset("k401ksubs")
k401ksubs_sub = @rsubset(k401ksubs, :fsize == 1)
skew(k401ksubs_sub.inc)
# here we get a slightly different score as library function assumes
# pupulation and zscore sample estimates
skewness(k401ksubs_sub.inc)
skew(log.(k401ksubs_sub.inc))
plot(histogram(k401ksubs_sub.inc; label="inc"),
     histogram(log.(k401ksubs_sub.inc); label="log(inc)"))

bwght2 = get_dataset("bwght2")
skew(bwght2.bwght)
skew(log.(bwght2.bwght))
plot(histogram(bwght2.bwght; label="bwght"),
     histogram(log.(bwght2.bwght); label="log(bwght)"))

# Problem C5

htv = get_dataset("htv")
combine(groupby(htv, :educ; sort=true), proprow)
histogram(htv.educ; normalize=:pdf, label="empirical")
plot!(x -> pdf(Normal(mean(htv.educ), std(htv.educ)), x);
      label="normal", linewidth=3)

educ_multi = lm(@formula(educ ~ motheduc + fatheduc + abil + abil^2), htv)
u = residuals(educ_multi)
histogram(u; normalize=:pdf, label="residuals")
plot!(x -> pdf(Normal(mean(u), std(u)), x);
      label="normal", linewidth=3)
scatter(predict(educ_multi), u, label="u")

