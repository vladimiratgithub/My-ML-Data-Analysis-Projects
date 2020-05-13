# Measurements of the an outocome of 70 successive chemical experiments.

yields <- read.table("yielnds.dat", header = FALSE)
t.yields <- ts(yields[, 1])

# Check the dependence and look for level of AR(p)
plot(t.yields)  # to check stationarity
acf(t.yields, na.action=na.pass, ylim=c(-1,1)) # checking the decay with increasing k= lag
pacf(t.yields, na.action=na.pass, ylim=c(-1,1))  # check the order of p to be used in AR(p)


# Use Yule-Walker to estimate parameters of AR(p)
r.yw <- ar(t.yields, method="yw", order.max=1)
str(r.yw)
print(r.yw)
