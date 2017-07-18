#=============================================================================================================
#Unsupervised Learning Demonstratio of 'TENNESSEE EASTMAN PROCESS'
#=============================================================================================================

#1 - PCA / LDA for Quality Data Analysis:
    #a) Building a PCA Model on the normal case data
    #b) Establish T2 and Q control limits on Quality Data
    #c) Use of LDA to study normal and disturbance data cases

#author: Dhineshkumar
#=============================================================================================================
#=============================================================================================================
#1.a) Building a PCA Modl on the normal case data
#=============================================================================================================

dim(d00)

d00.new = d00[seq(1, nrow(d00), 5), ]       ## Selecting every 5th row from d00 data (Data Redundancy)
qualityd00=d00.new[,37:41]                  ## Extracting Quality variables data for PCA

## Scaling the train data to zero mean and unit variance 
qualityd00_mean=apply(qualityd00,2,mean)
qualityd00_sd=apply(qualityd00,2,sd)

## Finding the Principal Components
qualityd00_scaled=scale(qualityd00, scale = T, center = T)
pr.out=prcomp(qualityd00_scaled)
plot(pr.out)
pr.out

## Finding Proportion of Variance Explained
pr.var=pr.out$sdev ^2
pve=pr.var/sum(pr.var)
pve
plot(pve,xlab="Principal component",ylab = "proportion of variance explaned",ylim=c(0,1),type = 'b')


#=============================================================================================================
#1.a)contd.. SPE and T2 Statistics on Quality Data
#=============================================================================================================

dim(d00)

d00.n = d00[seq(1,nrow(d00),5),]
dim(d00.n)

quality.n=d00.n[,37:41]
dim(quality.n)

quality.n_scaled=sweep(quality.n,2,qualityd00_mean)
dim(quality.n_scaled)

quality.n_scaled=sweep(quality.n_scaled,2,qualityd00_sd,FUN = "/")
N_test6=dim(quality.n_scaled)[1]

m6=dim(quality.n_scaled)[2]
quality.n_scaled=matrix(unlist(quality.n_scaled),nrow = N_test6, ncol = m6)

## SPE Statistics

P6=pr.out$rotation[,1:5]
Phi_Q6=diag(5)-P6%*%t(P6)
Q6=rep(0,N_test6)

for(i in seq(N_test6)){
  Q6[i]=t(quality.n_scaled[i,])%*%Phi_Q6%*%quality.n_scaled[i,]
}

plot(Q6, type="l", lty = 1, ylab = "SPE Statistic for Normal Data", col = "blue")
lines(rep(SPElimit, N_test6), type = "l", lty = 2, col="red")
legend(10, -0.00000000000000001, c("SPE Statistic", "Control Limit"), col=c("blue", "red"), lty=1:2)
P6

## T2 Stastics

Phi.T2_quality.n=P6%*%diag(pr.var[1:5]^(-1),5)%*%t(P6)
T2_quality.n=rep(0,N_test6)

for(i in seq(N_test6)){
  T2_quality.n[i]=t(quality.n_scaled[i,])%*%Phi.T2_quality.n%*%quality.n_scaled[i,]
}

plot(T2_quality.n, type = "l", lty = 1, ylab = "T2 Statistic for Normal Data", col = "blue", xlim= c(0,100), ylim=c(0,15))
lines(rep(T2limit,N_test6),type="l",lty=2, col="red")
legend(10,14,c("T2 Statistic for Normal Data","Control Limit"),col=c("blue","red"),lty=1:2)

#=============================================================================================================
#1.b) Establish T2 and Q control limits on Quality Data
#=============================================================================================================

## SPE Statistics
qualityd00_mean
theta1=sum(pr.var[5:5])       ## Calculate Pm i=l+1 ??i, which is ??1
theta2=sum(pr.var[5:5]^2)     ## Calculate Pm i=l+1 ??2i, which is ??2
g=theta2/theta1
h= theta1 ^2/theta2
SPElimit=g*qchisq(0.95, df=h)
SPElimit

## T2 Statistics
T2limit=qchisq(0.95, df=5) 
T2limit

#=============================================================================================================
#1.b)contd.. SPE and T2 Statistics on IDV02 Quality Data
#=============================================================================================================

dim(d02_te)

d02_te.New = d02_te[seq(1,nrow(d02_te),5),]
dim(d02_te.new)
IDV02=d02_te.new[,37:41]
dim(IDV02)

IDV02_scaled=sweep(IDV02,2,qualityd00_mean) ## Subtract each column with the mean calculated from each column of the train data
dim(IDV02_scaled)

IDV02_scaled=sweep(IDV02_scaled,2,qualityd00_sd,FUN = "/") ##Divide each column with the standard deviation calculated from each column of the train data
N_test1=dim(IDV02_scaled)[1]

m1=dim(IDV02_scaled)[2] ##Number of variables
IDV02_scaled=matrix(unlist(IDV02_scaled),nrow = N_test1, ncol = m1) ##Converting to double type

## SPE Statistics

P1=pr.out$rotation[,1:5]  ##Loading matrix, and in this case, the number of principal component is 5
Phi_Q1=diag(5)-P1%*%t(P1) ## ??Q = <I ??? PP>
Q1=rep(0,N_test1)         ##Create a N_test1 × 1 vector with all zeros

for(i in seq(N_test1)){
  Q1[i]=t(IDV02_scaled[i,])%*%Phi_Q1%*%IDV02_scaled[i,]
}

plot(Q1, type="l", lty = 1, ylab = "SPE Statistic", col = "blue")
lines(rep(SPElimit, N_test1), type = "l", lty = 2, col="red")
legend(10, 0.4, c("SPE Statistic", "Control Limit"), col=c("blue", "red"), lty=1:2)
P1

## T2 Statistics

Phi.T2_IDV02=P1%*%diag(pr.var[1:5]^(-1),5)%*%t(P1)
T2_IDV02=rep(0,N_test1)

for(i in seq(N_test1)){
  T2_IDV02[i]=t(IDV02_scaled[i,])%*%Phi.T2_IDV02%*%IDV02_scaled[i,]
}
plot(T2_IDV02, type = "l", lty = 1, ylab = "T2 Statistic for IDV02", col = "blue", xlim= c(0,200), ylim=c(0,60))
lines(rep(T2limit,N_test1),type="l",lty=2, col="red")
legend(10,50,c("T2 Statistic for IDV02","Control Limit"),col=c("blue","red"),lty=1:2)

#=============================================================================================================
#1.c) LDA on IDV02 Quality Data
#=============================================================================================================

n_d00_scaled=dim(qualityd00_scaled)[1] # Scale the quality data
N1_LDA=dim(IDV02_scaled)[1]
f_indicesIDV02=rep(0,N1_LDA)
n_indicesIDV02=rep(0,N1_LDA)
for(i in seq(N1_LDA)){
  if(T2_IDV02[i]>T2limit){
    f_indicesIDV02[i]=i
  }else{
    n_indicesIDV02[i]=i    
  }
}
fault_quality_IDV02=data.frame(IDV02_scaled[f_indicesIDV02,])
normal_quality_IDV02=data.frame(IDV02_scaled[n_indicesIDV02,])

names(fault_quality_IDV02)
names(normal_quality_IDV02)

## Changing names of the variables.

library(plyr)
fault_quality_IDV02 = rename(fault_quality_IDV02, c("X1"="X37", "X2"="X38", "X3"="X39","X4"="X40","X5"="X41"))
normal_quality_IDV02 = rename(normal_quality_IDV02, c("X1"="X37", "X2"="X38", "X3"="X39","X4"="X40","X5"="X41"))
names(fault_quality_IDV02)
names(normal_quality_IDV02)

## Concatenation of vectors

no_f=dim(fault_quality_IDV02)[1]
no_n=dim(normal_quality_IDV02)[1]
no_var1=n_d00_scaled+no_n

f1=rep(1,no_var1)
f1=data.frame(f1)
f2=rep(2,no_f)
f2=data.frame(f2)

LDA_IDV02= rbind(qualityd00_scaled,normal_quality_IDV02,fault_quality_IDV02)
dim(LDA_IDV02)

## Performing LDA

f1=rename(f1,c("f1"="Vard00"))
f2=rename(f2,c("f2"="Vard00"))
f3=rbind(f1,f2)
names(f3)
LDA_IDV02_data=data.frame(LDA_IDV02,f3)
dim(LDA_IDV02_data)
names(LDA_IDV02_data)
library (MASS)
lda.fit_IDV02=lda(Vard00 ~ X37+X38+X39+X40+X41,data=LDA_IDV02_data)
lda.fit_IDV02
plot(lda.fit_IDV02)


lda.fit.values<-predict(lda.fit,LDA_IDV02_data)

#=============================================================================================================
#1.b)contd.. SPE and T2 Statistics on IDV04 Quality Data (A different dataset)
#=============================================================================================================

##//d04_te=t(d04_te)
dim(d04_te)

d04_te.new = d04_te[seq(1,nrow(d04_te),5),]
dim(d04_te.new)

IDV04=d04_te.new[,37:41]
dim(IDV04)

IDV04_scaled=sweep(IDV04,2,qualityd00_mean)
dim(IDV04_scaled)

IDV04_scaled=sweep(IDV04_scaled,2,qualityd00_sd,FUN = "/")
N_test2=dim(IDV04_scaled)[1]

m2=dim(IDV04_scaled)[2]
IDV04_scaled=matrix(unlist(IDV04_scaled),nrow = N_test2, ncol = m2)

## SPE Statistics

P2=pr.out$rotation[,1:5]
Phi_Q2=diag(5)-P2%*%t(P2)
Q2=rep(0,N_test2)

for(i in seq(N_test2)){
  Q2[i]=t(IDV04_scaled[i,])%*%Phi_Q2%*%IDV04_scaled[i,]
}

plot(Q2, type="l", lty = 1, ylab = "SPE Statistic", col = "blue")
lines(rep(SPElimit, N_test2), type = "l", lty = 2, col="red")
legend(10, 0.00000000000000000000000000001, c("SPE Statistic", "Control Limit"), col=c("blue", "red"), lty=1:2)
P2

## T2 Statistics

Phi.T2_IDV04=P2%*%diag(pr.var[1:5]^(-1),5)%*%t(P2)
T2_IDV04=rep(0,N_test2)

for(i in seq(N_test2)){
  T2_IDV04[i]=t(IDV04_scaled[i,])%*%Phi.T2_IDV04%*%IDV04_scaled[i,]
}

plot(T2_IDV04, type = "l", lty = 1, ylab = "T2 Statistic for IDV04", col = "blue", xlim= c(0,200), ylim=c(0,30))
lines(rep(T2limit,N_test2),type="l",lty=2, col="red")
legend(10,28,c("T2 Statistic for IDV04","Control Limit"),col=c("blue","red"),lty=1:2)

#=============================================================================================================
#1.c)contd.. LDA on IDV04 Quality Data
#=============================================================================================================

n_d00_scaled=dim(qualityd00_scaled)[1]
N1_LDA=dim(IDV04_scaled)[1]
f_indicesIDV04=rep(0,N1_LDA)
n_indicesIDV04=rep(0,N1_LDA)

for(i in seq(N1_LDA)){
  if(T2_IDV04[i]>T2limit){
    f_indicesIDV04[i]=i
  }else{
    n_indicesIDV04[i]=i    
  }
}

fault_quality_IDV04=data.frame(IDV04_scaled[f_indicesIDV04,])
normal_quality_IDV04=data.frame(IDV04_scaled[n_indicesIDV04,])

names(fault_quality_IDV04)
names(normal_quality_IDV04)

## changing names of the variables

library(plyr)
fault_quality_IDV04 = rename(fault_quality_IDV04, c("X1"="X37", "X2"="X38", "X3"="X39","X4"="X40","X5"="X41"))
normal_quality_IDV04 = rename(normal_quality_IDV04, c("X1"="X37", "X2"="X38", "X3"="X39","X4"="X40","X5"="X41"))
names(fault_quality_IDV04)
names(normal_quality_IDV04)

## Concatenation of vectors

no_f=dim(fault_quality_IDV04)[1]
no_n=dim(normal_quality_IDV04)[1]
no_var1=n_d00_scaled+no_n

f1=rep(1,no_var1)
f1=data.frame(f1)
f2=rep(2,no_f)
f2=data.frame(f2)

LDA_IDV04= rbind(qualityd00_scaled,normal_quality_IDV04,fault_quality_IDV04)
dim(LDA_IDV04)

## Performing LDA

f1=rename(f1,c("f1"="Vard00"))
f2=rename(f2,c("f2"="Vard00"))
f3=rbind(f1,f2)
names(f3)

LDA_IDV04_data=data.frame(LDA_IDV04,f3)
dim(LDA_IDV04_data)
names(LDA_IDV04_data)

library (MASS)
lda.fit_IDV04=lda(Vard00 ~ X37+X38+X39+X40+X41,data=LDA_IDV04_data)
lda.fit_IDV04
plot(lda.fit_IDV04)

#=============================================================================================================
