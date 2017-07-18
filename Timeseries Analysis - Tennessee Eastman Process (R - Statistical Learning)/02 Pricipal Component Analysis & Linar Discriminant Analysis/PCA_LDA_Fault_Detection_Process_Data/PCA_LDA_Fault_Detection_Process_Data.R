#=============================================================================================================
#Unsupervised Learning Demonstratio of 'TENNESSEE EASTMAN PROCESS'
#=============================================================================================================

#2 - PCA / LDA for Process Data Analysis:
    #a) Building a PCA Modl based on the normal Process data
    #b) Establish T2 and Q control limits on Process Data
    #c) Calculation of False Alarm Rates and Missed Detection Rates
    #d) Use of LDA to study normal and disturbance data cases 

#author: Dhineshkumar
#=============================================================================================================
#=============================================================================================================
#1.a) Building a PCA Modl on the normal case data
#=============================================================================================================

dim(d00)
n.d00.new = d00[seq(1, nrow(d00), 5), ]                   ## Selecting every 5th row from d00 data (Data Redundancy)
normal.d00=data.frame(n.d00.new[,1:22],n.d00.new[,42:52]) ## Extracting Process variables data for PCA
dim(normal.d00)

which(apply(normal.d00, 2, var)==0)   ## Identifying columns with zero variance
n.normal.d00=normal.d00[ , apply(normal.d00, 2, var) != 0]
dim(n.normal.d00)

## Scaling the train data to zero mean and unit variance 
normald00_mean=apply(n.normal.d00,2,mean) ## Scaling the train data to zero mean and unit variance 
normald00_sd=apply(n.normal.d00,2,sd)
normald00_scaled=scale(n.normal.d00, scale = T, center = T)
dim(normald00_scaled)

## Finding the Principal Components
n.pr.out=prcomp(normald00_scaled)
plot(n.pr.out)
n.pr.out
n.pr.var=n.pr.out$sdev ^2

## Calculating Proportion of Variance Explained
n.pve=n.pr.var/sum(n.pr.var)
n.pve
plot(n.pve,xlab="Principal component",ylab = "Proportion of variance explaned",ylim=c(0,1),type = 'b')
cumsum(n.pve)
plot(cumsum(n.pve),xlab="Principal component",ylab = "Cumulative PVE",ylim=c(0,1),type = 'b')

## SPE Control Limit Determination
normald00_mean
n.theta1=sum(n.pr.var[20:32])
n.theta2=sum(n.pr.var[20:32]^2)
n.g=n.theta2/n.theta1
n.h= n.theta1 ^2/n.theta2
n.SPElimit=n.g*qchisq(0.95, df=n.h)
n.SPElimit

## T2 Control Limit Determination
n.T2limit=qchisq(0.95, df=19) 
n.T2limit

#=============================================================================================================
#From the value of pve and the plot, we can see the proportion of variance explained by various principal components.
#We select the number of Principal Components based on the number of components whose cumulative PVE is more than 95%. Therefore, the number of PCs selected are 19.
#For the further calculations, variables 'm' (total number of PC's) = 32 and 'l' (no. PC's selected) = 19
#In this case both SPE statistics and T2 statistics are significant. However we have used T2 Statistics for further data analytics. 
#=============================================================================================================

#=============================================================================================================
#1.b) SPE and T2 Statistics on IDV02 Process Data
#=============================================================================================================

dim(d02_te)
n.d02_te.new = d02_te[seq(1, nrow(d02_te), 5), ]
n.IDV02=data.frame(n.d02_te.new[,1:22],n.d02_te.new[,42:52])
dim(n.IDV02)

which(apply(n.IDV02, 2, var)==0) ## Identifying columns with zero variance
n.n.IDV02=n.IDV02[ , apply(n.IDV02, 2, var) != 0]
dim(n.n.IDV02)

n.n.IDV02_scaled=sweep(n.n.IDV02,2,normald00_mean)
dim(n.n.IDV02_scaled)
n.n.IDV02_scaled=sweep(n.n.IDV02_scaled,2,normald00_sd,FUN = "/") ##Divide each column with the standard deviation calculated from each column of the train data

N_test7=dim(n.n.IDV02_scaled)[1]
m7=dim(n.n.IDV02_scaled)[2]       ##Number of variables
n.n.IDV02_scaled=matrix(unlist(n.n.IDV02_scaled),nrow = N_test7, ncol = m7)


## SPE Statistics

P21=n.pr.out$rotation[,1:19]      ## Loading matrix, and in this case, the number of principal component is 19
Phi_Q21=diag(32)-P21%*%t(P21)
Q21=rep(0,N_test7)
for(i in seq(N_test7)){
  Q21[i]=t(n.n.IDV02_scaled[i,])%*%Phi_Q21%*%n.n.IDV02_scaled[i,]
}
plot(Q21, type="l", lty = 1, ylab = "SPE Statistic for IDV02", col = "blue")
lines(rep(n.SPElimit, N_test7), type = "l", lty = 2, col="red")
legend(5, 80, c("SPE Statistic", "Control Limit"), col=c("blue", "red"), lty=1:2)

## T2 Statistics

Phi.T2_n.n.IDV02=P21%*%diag(n.pr.var[1:19]^(-1))%*%t(P21)
T2_n.n.IDV02=rep(0,N_test7)
for(i in seq(N_test7)){
  T2_n.n.IDV02[i]=t(n.n.IDV02_scaled[i,]%*%Phi.T2_n.n.IDV02%*%n.n.IDV02_scaled[i,])
}
plot(T2_n.n.IDV02, type = "l", lty = 1, ylab = "T2 Statistic for IDV02", col = "blue", xlim= c(0,200), ylim=c(0,3000))
lines(rep(n.T2limit,N_test7),type="l",lty=2, col="red")
legend(10,3000,c("T2 Statistic for IDV02","Control Limit"),col=c("blue","red"),lty=1:2)

#=============================================================================================================
#1.c) False Alarm Rate and Missed Detection Rate calcs for IDV02 Process Data
#=============================================================================================================

# False Alarm Rate

i=0

d_quality_T2_IDV02=dim(data.frame(T2_IDV02))[1]
d_normal_T2_IDV02=dim(data.frame(T2_n.n.IDV02))[1]
d_quality_T2_IDV02

FAR_NOT_FAULT_BUT_T2_GREATER_IDV02 = 0
FAR_NOT_FAULT_IN_PROCESS_IDV02 = 0

for(i in seq(d_normal_T2_IDV02)){
  if(T2_IDV02[i] > T2limit & T2_n.n.IDV02[i] < n.T2limit){
    FAR_NOT_FAULT_BUT_T2_GREATER_IDV02 = FAR_NOT_FAULT_BUT_T2_GREATER_IDV02 +1
  }
  if(T2_n.n.IDV02[i] <= n.T2limit){
    FAR_NOT_FAULT_IN_PROCESS_IDV02 = FAR_NOT_FAULT_IN_PROCESS_IDV02 +1
  }
}

FAR_NOT_FAULT_BUT_T2_GREATER_IDV02
FAR_NOT_FAULT_IN_PROCESS_IDV02
FAR_IDV02 = FAR_NOT_FAULT_BUT_T2_GREATER_IDV02/FAR_NOT_FAULT_IN_PROCESS_IDV02
FAR_IDV02         ## Result obtained: 16.12%

# Missed Detection  Rate

i=0
MDR_FAULT_BUT_T2_LESS_IDV02 = 0
MDR_FAULT_IN_PROCESS_IDV02 = 0

for(i in seq(d_normal_T2_IDV02)){
  if(T2_n.n.IDV02[i] > n.T2limit & T2_IDV02[i] < T2limit){
    MDR_FAULT_BUT_T2_LESS_IDV02 = MDR_FAULT_BUT_T2_LESS_IDV02 +1
  }
  if(T2_n.n.IDV02[i] >= n.T2limit){
    MDR_FAULT_IN_PROCESS_IDV02 = MDR_FAULT_IN_PROCESS_IDV02 +1
  }
}

MDR_FAULT_BUT_T2_LESS_IDV02
MDR_FAULT_IN_PROCESS_IDV02
MDR_IDV02 = MDR_FAULT_BUT_T2_LESS_IDV02/MDR_FAULT_IN_PROCESS_IDV02
MDR_IDV02        ##Result Obtained 18.01%

#=============================================================================================================
#1.d) Linear Discriminant Analysis on IDV02 Process Data
#=============================================================================================================

# Scale the quality data

n21_d00_scaled=dim(normald00_scaled)[1]
N21_LDA=dim(IDV02_scaled)[1]
f21_indicesIDV02=rep(0,N21_LDA)
n21_indicesIDV02=rep(0,N21_LDA)

for(i in seq(N21_LDA)){
  if(T2_n.n.IDV02[i]>n.T2limit){
    f21_indicesIDV02[i]=i
  }else{
    n21_indicesIDV02[i]=i    
  }
}

fault21_normal_IDV02=data.frame(n.n.IDV02_scaled[f21_indicesIDV02,])
normal21_normal_IDV02=data.frame(n.n.IDV02_scaled[n21_indicesIDV02,])

names(fault21_normal_IDV02)
names(normal21_normal_IDV02)

## changing names of the variables

library(plyr)
fault21_normal_IDV02 = rename(fault21_normal_IDV02, c("X9"="X10","X10"="X11","X11"="X12","X12"="X13","X13"="X14","X14"="X15","X15"="X16","X16"="X17","X17"="X18","X18"="X19","X19"="X20","X20"="X21","X21"="X22","X22"="X42","X23"="X43", "X24"="X44","X25"="X45","X26"="X46","X27"="X47","X28"="X48","X29"="X49","X30"="X50","X31"="X51","X32"="X52"))
normal21_normal_IDV02 = rename(normal21_normal_IDV02, c("X9"="X10","X10"="X11","X11"="X12","X12"="X13","X13"="X14","X14"="X15","X15"="X16","X16"="X17","X17"="X18","X18"="X19","X19"="X20","X20"="X21","X21"="X22","X22"="X42","X23"="X43", "X24"="X44","X25"="X45","X26"="X46","X27"="X47","X28"="X48","X29"="X49","X30"="X50","X31"="X51","X32"="X52"))
names(fault21_normal_IDV02)
names(normal21_normal_IDV02)

## Concatenation of vectors

no_f_IDV02=dim(fault21_normal_IDV02)[1]
no_n_IDV02=dim(normal21_normal_IDV02)[1]
no_var21=n21_d00_scaled+no_n_IDV02

f1_IDV02=rep(1,no_var21)
f1_IDV02=data.frame(f1_IDV02)
f2_IDV02=rep(2,no_f_IDV02)
f2_IDV02=data.frame(f2_IDV02)

LDA21_IDV02= rbind(normald00_scaled,normal21_normal_IDV02,fault21_normal_IDV02)
dim(LDA21_IDV02)
##dim(f3_IDV02)

## Performing LDA

f1_name_IDV02 = names(f1_IDV02)[1]
f2_name_IDV02 = names(f2_IDV02)[2]
##dim(f4_IDV02)

f1_IDV02=rename(f1_IDV02,c("f1_IDV02"="Vard00_IDV02"))
f2_IDV02=rename(f2_IDV02,c("f2_IDV02"="Vard00_IDV02"))
f3_IDV02=rbind(f1_IDV02,f2_IDV02)
names(f3_IDV02)

LDA21_IDV02_data=data.frame(LDA21_IDV02,f3_IDV02)
dim(LDA21_IDV02_data)
names(LDA21_IDV02_data)

library (MASS)
lda.fit_21=lda(Vard00_IDV02 ~ X1+X2+X3+X4+X5+X6+X7+X8+X10+X11+X12+X13+X14+X15+X6+X17+X18+X19+X20+X21+X22+X42+X43+X44+X45+X46+X47+X48+X49+X50+X51+X52,data=LDA21_IDV02_data)
lda.fit_21

plot(lda.fit_21,ylim=c(0,1))

#=============================================================================================================
#1.b)cont.. SPE and T2 Statistics on IDV04 Process Data
#=============================================================================================================


dim(d04_te)
n.d04_te.new = d04_te[seq(1, nrow(d04_te), 5), ]
n.IDV04=data.frame(n.d04_te.new[,1:22],n.d04_te.new[,42:52])
n.n.IDV04=n.IDV04
dim(n.IDV04)

which(apply(n.IDV04, 2, var)==0)
n.n.IDV04=n.IDV04[ , apply(n.IDV04, 2, var) != 0]
dim(n.n.IDV04)

n.n.IDV04 <- n.n.IDV04[-c(9)]
dim(n.n.IDV04)

n.n.IDV04_scaled=sweep(n.n.IDV04,2,normald00_mean)
dim(n.n.IDV04_scaled)
n.n.IDV04_scaled=sweep(n.n.IDV04_scaled,2,normald00_sd,FUN = "/")
N_test8=dim(n.n.IDV04_scaled)[1]
m8=dim(n.n.IDV04_scaled)[2]
n.n.IDV04_scaled=matrix(unlist(n.n.IDV04_scaled),nrow = N_test8, ncol = m8)

## SPE Statistics

P22=n.pr.out$rotation[,1:19]
dim(P22)
Phi_Q22=diag(32)-P22%*%t(P22)
Q22=rep(0,N_test8)

for(i in seq(N_test8)){
  Q22[i]=t(n.n.IDV04_scaled[i,])%*%Phi_Q22%*%n.n.IDV04_scaled[i,]
}

plot(Q22, type="l", lty = 1, ylab = "SPE Statistic for IDV04", col = "blue")
lines(rep(n.SPElimit, N_test8), type = "l", lty = 2, col="red")
legend(10,25, c("SPE Statistic", "Control Limit"), col=c("blue", "red"), lty=1:2)

## T2 Statistics

Phi.T2_n.n.IDV04=P22%*%diag(n.pr.var[1:19]^(-1))%*%t(P22)
T2_n.n.IDV04=rep(0,N_test8)

for(i in seq(N_test8)){
  T2_n.n.IDV04[i]=t(n.n.IDV04_scaled[i,]%*%Phi.T2_n.n.IDV04%*%n.n.IDV04_scaled[i,])
}

plot(T2_n.n.IDV04, type = "l", lty = 1, ylab = "T2 Statistic for IDV04", col = "blue", xlim= c(0,200), ylim=c(0,150))
lines(rep(n.T2limit,N_test8),type="l",lty=2, col="red")
legend(10,150,c("T2 Statistic for IDV04","Control Limit"),col=c("blue","red"),lty=1:2)

#=============================================================================================================
#1.c)contd... False Alarm Rate and Missed Detection Rate calcs for IDV04 Process Data
#=============================================================================================================

# False Alarm Rate

i=0
d_quality_T2_IDV04=dim(data.frame(T2_IDV04))[1]
d_normal_T2_IDV04=dim(data.frame(T2_n.n.IDV04))[1]
d_quality_T2_IDV04
FAR_NOT_FAULT_BUT_T2_GREATER_IDV04 = 0
FAR_NOT_FAULT_IN_PROCESS_IDV04 = 0

for(i in seq(d_normal_T2_IDV04)){
  if(T2_IDV04[i] > T2limit & T2_n.n.IDV04[i] < n.T2limit){
    FAR_NOT_FAULT_BUT_T2_GREATER_IDV04 = FAR_NOT_FAULT_BUT_T2_GREATER_IDV04 +1
  }
  if(T2_n.n.IDV04[i] <= n.T2limit){
    FAR_NOT_FAULT_IN_PROCESS_IDV04 = FAR_NOT_FAULT_IN_PROCESS_IDV04 +1
  }
}

FAR_NOT_FAULT_BUT_T2_GREATER_IDV04
FAR_NOT_FAULT_IN_PROCESS_IDV04
FAR_IDV04 = FAR_NOT_FAULT_BUT_T2_GREATER_IDV04/FAR_NOT_FAULT_IN_PROCESS_IDV04
FAR_IDV04    ##Calculate Reult = 14.28%

## Missed Detection Rate

i=0
MDR_FAULT_BUT_T2_LESS_IDV04 = 0
MDR_FAULT_IN_PROCESS_IDV04 = 0

for(i in seq(d_normal_T2_IDV04)){
  if(T2_n.n.IDV04[i] > n.T2limit & T2_IDV04[i] < T2limit){
    MDR_FAULT_BUT_T2_LESS_IDV04 = MDR_FAULT_BUT_T2_LESS_IDV04 +1
  }
  if(T2_n.n.IDV04[i] >= n.T2limit){
    MDR_FAULT_IN_PROCESS_IDV04 = MDR_FAULT_IN_PROCESS_IDV04 +1
  }
}

MDR_FAULT_BUT_T2_LESS_IDV04
MDR_FAULT_IN_PROCESS_IDV04
MDR_IDV04 = MDR_FAULT_BUT_T2_LESS_IDV04/MDR_FAULT_IN_PROCESS_IDV04
MDR_IDV04  ##Calculated MDR = 89.63%

#=============================================================================================================
#1.d) Linear Discriminant Analysis on IDV04 Process Data
#=============================================================================================================


n22_d00_scaled=dim(normald00_scaled)[1]
N22_LDA=dim(IDV04_scaled)[1]
f22_indicesIDV04=rep(0,N22_LDA)
n22_indicesIDV04=rep(0,N22_LDA)

for(i in seq(N22_LDA)){
  if(T2_n.n.IDV04[i]>n.T2limit){
    f22_indicesIDV04[i]=i
  }else{
    n22_indicesIDV04[i]=i    
  }
}

fault22_normal_IDV04=data.frame(n.n.IDV04_scaled[f22_indicesIDV04,])
normal22_normal_IDV04=data.frame(n.n.IDV04_scaled[n22_indicesIDV04,])

names(fault22_normal_IDV04)
names(normal22_normal_IDV04)

## changing names of the variables

library(plyr)
fault22_normal_IDV04 = rename(fault22_normal_IDV04, c("X9"="X10","X10"="X11","X11"="X12","X12"="X13","X13"="X14","X14"="X15","X15"="X16","X16"="X17","X17"="X18","X18"="X19","X19"="X20","X20"="X21","X21"="X22","X22"="X42","X23"="X43", "X24"="X44","X25"="X45","X26"="X46","X27"="X47","X28"="X48","X29"="X49","X30"="X50","X31"="X51","X32"="X52"))
normal22_normal_IDV04 = rename(normal22_normal_IDV04, c("X9"="X10","X10"="X11","X11"="X12","X12"="X13","X13"="X14","X14"="X15","X15"="X16","X16"="X17","X17"="X18","X18"="X19","X19"="X20","X20"="X21","X21"="X22","X22"="X42","X23"="X43", "X24"="X44","X25"="X45","X26"="X46","X27"="X47","X28"="X48","X29"="X49","X30"="X50","X31"="X51","X32"="X52"))
names(fault22_normal_IDV04)
names(normal22_normal_IDV04)

## Concatenation of vectors

no_f_IDV04=dim(fault22_normal_IDV04)[1]
no_n_IDV04=dim(normal22_normal_IDV04)[1]
no_var22=n22_d00_scaled+no_n_IDV04

f1_IDV04=rep(1,no_var22)
f1_IDV04=data.frame(f1_IDV04)
f2_IDV04=rep(2,no_f_IDV04)
f2_IDV04=data.frame(f2_IDV04)

LDA22_IDV04= rbind(normald00_scaled,normal22_normal_IDV04,fault22_normal_IDV04)
dim(LDA22_IDV04)
##dim(f3)

## Performing LDA

f1_name_IDV04 = names(f1_IDV04)[1]
f2_name_IDV04 = names(f2_IDV04)[2]
##dim(f4)

f1_IDV04=rename(f1_IDV04,c("f1_IDV04"="Vard00_IDV04"))
f2_IDV04=rename(f2_IDV04,c("f2_IDV04"="Vard00_IDV04"))
f3_IDV04=rbind(f1_IDV04,f2_IDV04)
names(f3_IDV04)

LDA22_IDV04_data=data.frame(LDA22_IDV04,f3_IDV04)
dim(LDA22_IDV04_data)
names(LDA22_IDV04_data)

library (MASS)
lda.fit_22=lda(Vard00_IDV04 ~ X1+X2+X3+X4+X5+X6+X7+X8+X10+X11+X12+X13+X14+X15+X6+X17+X18+X19+X20+X21+X22+X42+X43+X44+X45+X46+X47+X48+X49+X50+X51+X52,data=LDA22_IDV04_data)
lda.fit_22

plot(lda.fit_22,ylim=c(0,1))

#=============================================================================================================
