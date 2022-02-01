import skfuzzy as fuzz
import numpy as np
import skfuzzy.membership as mf
import matplotlib.pyplot as  plt

def prod(a,b):
    return a*b

input_market=int(input("input for market value: "))
input_location=int(input("input for location: "))
input_asset=int(input("input for asset: "))
input_income=int(input("input for income: "))
input_interest=int(input("input for interest: "))

Market_value=np.arange(0,1001)
Location=np.arange(0,11,0.1)
House=np.arange(0,11)

Asset=np.arange(0,1001,1)
Income=np.arange(0,101,1)
Applicant=np.arange(0,11,0.1)


Interest =np.arange(0,11,0.1)

Credit=np.arange(0,501,1)


##########House#########
Market_value_low=mf.trapmf(Market_value, [0,0,50,100])
Market_value_medium=mf.trapmf(Market_value, [50,100,200,250])
Market_value_high=mf.trapmf(Market_value,  [200,300,650,850])
Market_value_very_high=mf.trapmf(Market_value, [650,850,1000,1000])

Location_bad=mf.trapmf(Location, [0,0,1.5,4])
Location_fair=mf.trapmf(Location,  [2.5,5,6,8.5])
Location_excellent=mf.trapmf(Location, [6,8.5,10,10])


House_very_low=mf.trimf(House, [0,0,3])
House_low=mf.trimf(House, [0,3,6])
House_medium=mf.trimf(House, [2,5,8])
House_high=mf.trimf(House, [4,7,10])
House_very_high=mf.trimf(House, [7,10,10])

fig ,(ax0,ax1,ax2)=plt.subplots(nrows=3,figsize=(6,10))
ax0.plot(Market_value,Market_value_low,'r',linewidth=2,label='low')
ax0.plot(Market_value,Market_value_medium,'g',linewidth=2,label='medium')
ax0.plot(Market_value,Market_value_high,'b',linewidth=2,label='high')
ax0.plot(Market_value,Market_value_very_high,'y',linewidth=2,label='very_high')
ax0.set_title('Market_Value')
ax0.legend()

ax1.plot(Location,Location_bad,'r',linewidth=2,label='bad')
ax1.plot(Location,Location_fair,'g',linewidth=2,label='fair')
ax1.plot(Location,Location_excellent,'b',linewidth=2,label='excellent')
ax1.set_title('Location')
ax1.legend()

ax2.plot(House,House_very_low,'r',linewidth=2,label='very low')
ax2.plot(House,House_low,'g',linewidth=2,label='low')
ax2.plot(House,House_medium,'b',linewidth=2,label='medium')
ax2.plot(House,House_high,'y',linewidth=2,label='high')
ax2.plot(House,House_very_high,'k',linewidth=2,label='very high')
ax2.set_title('House')
ax2.legend()

plt.tight_layout()

Market_fit_low=fuzz.interp_membership(Market_value,Market_value_low,input_market)
Market_fit_medium=fuzz.interp_membership(Market_value,Market_value_medium,input_market)
Market_fit_high=fuzz.interp_membership(Market_value,Market_value_high,input_market)
market_fit_very_high=fuzz.interp_membership(Market_value,Market_value_very_high,input_market)

Location_fit_bad=fuzz.interp_membership(Location,Location_bad,input_location)
Location_fit_fair=fuzz.interp_membership(Location,Location_fair,input_location)
Location_fit_excellent=fuzz.interp_membership(Location,Location_excellent,input_location)


rule1_house=prod(Market_fit_low,House_low)
rule2_house=prod(Location_fit_bad,House_low)
rule3_house=prod(np.fmin(Location_fit_bad,Market_fit_low),House_very_low)
rule4_house=prod(np.fmin(Location_fit_bad,Market_fit_medium),House_low)
rule5_house=prod(np.fmin(Location_fit_bad,Market_fit_high),House_medium)
rule6_house=prod(np.fmin(Location_fit_bad,market_fit_very_high),House_high)
rule7_house=prod(np.fmin(Location_fit_fair,Market_fit_low),House_low)
rule8_house=prod(np.fmin(Location_fit_fair,Market_fit_medium),House_medium)
rule9_house=prod(np.fmin(Location_fit_fair,Market_fit_high),House_high)
rule10_house=prod(np.fmin(Location_fit_fair,market_fit_very_high),House_very_high)
rule11_house=prod(np.fmin(Location_fit_excellent,Market_fit_low),House_medium)
rule12_house=prod(np.fmin(Location_fit_excellent,Market_fit_medium),House_high)
rule13_house=prod(np.fmin(Location_fit_excellent,Market_fit_high),House_very_high)
rule14_house=prod(np.fmin(Location_fit_excellent,market_fit_very_high),House_very_high)


out_house_very_low=rule3_house
out_house_low=np.fmax(np.fmax(rule1_house,rule2_house),np.fmax(rule4_house,rule7_house))
out_house_medium=np.fmax(np.fmax(rule5_house,rule8_house),rule11_house)
out_house_high=np.fmax(np.fmax(rule6_house,rule9_house),rule12_house)
out_house_very_high=np.fmax(np.fmax(rule10_house,rule14_house),rule13_house)


output_house=np.fmax(np.fmax(np.fmax(out_house_high,out_house_low),np.fmax(out_house_medium,out_house_very_low)),out_house_very_high)
defuzzified_house=fuzz.defuzz(House,output_house,'centroid')
result_house=fuzz.interp_membership(House,output_house,defuzzified_house)

House0=np.zeros_like(House)
fig, ax7 = plt.subplots(figsize=(7, 4))

ax7.plot(House,House_very_low,'b',linewidth=1,linestyle='--')
ax7.plot(House,House_low,'r',linewidth=1,linestyle='--')
ax7.plot(House,House_medium,'g',linewidth=1,linestyle='--')
ax7.plot(House,House_high,'y',linewidth=1,linestyle='--')
ax7.plot(House,House_very_high,'b',linewidth=1,linestyle='--')
ax7.fill_between(House,House0,output_house,facecolor='Blue',alpha=0.7)
ax7.plot([defuzzified_house,defuzzified_house],[0,result_house],'k',linewidth=2 ,alpha=0.9)
ax7.set_title('Girilen input değerlerine karşılık House için durulaştırılmış grafik')       
plt.tight_layout()
print(f"Girilen input değerlerine karşılık house için durulaştırılmış  değer:{round(defuzzified_house,2)}")  
 
#################House###########################




##########APPLICANT#####################

Asset_low=mf.trapmf(Asset, [0,0,0,150])
Asset_medium=mf.trapmf(Asset,[50,250,450,650])
Asset_high=mf.trapmf(Asset, [500,700,1000,1000])

Income_low=mf.trapmf(Income, [0,0,10,25])
Income_medium=mf.trapmf(Income, [15,35,35,55])
Income_high=mf.trapmf(Income,  [40,60,60,80])
Income_very_high=mf.trapmf(Income,[60,80,100,100])

Applicant_low=mf.trapmf(Applicant, [0,0,2,4])
Applicant_medium=mf.trapmf(Applicant,  [2,5,5,8])
Applicant_high=mf.trapmf(Applicant,  [6,8,10,10])

fig ,(ax8,ax9,ax10)=plt.subplots(nrows=3,figsize=(6,10))
ax8.plot(Asset,Asset_low,'r',linewidth=2,label='low')
ax8.plot(Asset,Asset_medium,'g',linewidth=2,label='medium')
ax8.plot(Asset,Asset_high,'b',linewidth=2,label='high')
ax8.set_title('Asset')
ax8.legend()

ax9.plot(Income,Income_low,'r',linewidth=2,label='low')
ax9.plot(Income,Income_medium,'g',linewidth=2,label='medium')
ax9.plot(Income,Income_high,'b',linewidth=2,label='high')
ax9.plot(Income,Income_very_high,'y',linewidth=2,label='very high')
ax9.set_title('Income')
ax9.legend()

ax10.plot(Applicant,Applicant_low,'r',linewidth=2,label='low')
ax10.plot(Applicant,Applicant_medium,'g',linewidth=2,label='medium')
ax10.plot(Applicant,Applicant_high,'b',linewidth=2,label='high')
ax10.set_title('Applicant')
ax10.legend()

plt.tight_layout()


Asset_fit_low=fuzz.interp_membership(Asset,Asset_low,input_asset)
Asset_fit_medium=fuzz.interp_membership(Asset,Asset_medium,input_asset)
Asset_fit_high=fuzz.interp_membership(Asset,Asset_high,input_asset)

Income_fit_low=fuzz.interp_membership(Income,Income_low,input_income)
Income_fit_medium=fuzz.interp_membership(Income,Income_medium,input_income)
Income_fit_high=fuzz.interp_membership(Income,Income_high,input_income)
Income_fit_very_high=fuzz.interp_membership(Income,Income_very_high,input_income)

rule1_applicant=prod(np.fmin(Asset_fit_low,Income_fit_low),Applicant_low)
rule2_applicant=prod(np.fmin(Asset_fit_low,Income_fit_medium),Applicant_low)
rule3_applicant=prod(np.fmin(Asset_fit_low,Income_fit_high),Applicant_medium)
rule4_applicant=prod(np.fmin(Asset_fit_low,Income_fit_very_high),Applicant_high)
rule5_applicant=prod(np.fmin(Asset_fit_medium,Income_fit_low),Applicant_low)
rule6_applicant=prod(np.fmin(Asset_fit_medium,Income_fit_medium),Applicant_medium)
rule7_applicant=prod(np.fmin(Asset_fit_medium,Income_fit_high),Applicant_high)
rule8_applicant=prod(np.fmin(Asset_fit_medium,Income_fit_very_high),Applicant_high)
rule9_applicant=prod(np.fmin(Asset_fit_high,Income_fit_low),Applicant_medium)
rule10_applicant=prod(np.fmin(Asset_fit_high,Income_fit_medium),Applicant_medium)
rule11_applicant=prod(np.fmin(Asset_fit_high,Income_fit_high),Applicant_high)
rule12_applicant=prod(np.fmin(Asset_fit_high,Income_fit_very_high),Applicant_high)


out_applicant_low=np.fmax(rule1_applicant,np.fmax(rule2_applicant,rule5_applicant))
out_applicant_medium=np.fmax(np.fmax(rule3_applicant,rule6_applicant),np.fmax(rule9_applicant,rule10_applicant))
out_applicant_high=np.fmax(np.fmax(np.fmax(rule4_applicant,rule7_applicant),np.fmax(rule8_applicant,rule11_applicant)),rule12_applicant)


output_applicant=np.fmax(np.fmax(out_applicant_low,out_applicant_medium),out_applicant_high)
defuzzified_applicant=fuzz.defuzz(Applicant,output_applicant,'centroid')
result_applicant=fuzz.interp_membership(Applicant,output_applicant,defuzzified_applicant)

Applicant0=np.zeros_like(Applicant)
fig, ax11=plt.subplots(figsize=(7,4))
ax11.plot(Applicant,Applicant_low,'r',linewidth=1,linestyle='--')
ax11.plot(Applicant,Applicant_medium,'g',linewidth=1,linestyle='--')
ax11.plot(Applicant,Applicant_high,'b',linewidth=1,linestyle='--')
ax11.fill_between(Applicant,Applicant0,output_applicant,facecolor='Blue',alpha=0.7)
ax11.plot([defuzzified_applicant,defuzzified_applicant],[0,result_applicant],'k',linewidth=2,alpha=0.9)
ax11.set_title('Girilen input değerlerine karşılık Applicant için durulaştırılmış grafik')
plt.tight_layout()
print(f"Girilen input değerlerine karşılık Applicant için durulaştırılmış  değer:{round(defuzzified_applicant,2)}")

###########################APPLICANT##########################

###########CREDIT#######################
Interest_low=mf.trapmf(Interest, [0,0,2,5])
Interest_medium=mf.trapmf(Interest, [2,4,6,8])
Interest_high=mf.trapmf(Interest, [6,8.5,10,10])

Credit_very_low=mf.trimf(Credit, [0,0,125])
Credit_low=mf.trimf(Credit,  [0,125,250])
Credit_medium=mf.trimf(Credit, [125,250,375] )
Credit_high=mf.trimf(Credit,  [250,375,500] )
Credit_very_high=mf.trimf(Credit, [375,500,500] )


fig ,(ax11,ax12)=plt.subplots(nrows=2,figsize=(6,10))

ax11.plot(Interest,Interest_low,'r',linewidth=2,label='low')
ax11.plot(Interest,Interest_medium,'g',linewidth=2,label='medium')
ax11.plot(Interest,Interest_high,'b',linewidth=2,label='high')
ax11.set_title('Interest')
ax11.legend()

ax12.plot(Credit,Credit_very_low,'r',linewidth=2,label='very low')
ax12.plot(Credit,Credit_low,'g',linewidth=2,label=' low')
ax12.plot(Credit,Credit_medium,'b',linewidth=2,label='medium')
ax12.plot(Credit,Credit_high,'black',linewidth=2,label='high')
ax12.plot(Credit,Credit_very_high,'y',linewidth=2,label='very high')
ax12.set_title('Credit')
ax12.legend()
plt.tight_layout()

Interest_fit_low=fuzz.interp_membership(Interest,Interest_low,input_interest)
Interest_fit_medium=fuzz.interp_membership(Interest,Interest_medium,input_interest)
Interest_fit_high=fuzz.interp_membership(Interest,Interest_high,input_interest)

House_fit_very_low=fuzz.interp_membership(House,House_very_low,defuzzified_house)
House_fit_low=fuzz.interp_membership(House,House_low,defuzzified_house)
House_fit_medium=fuzz.interp_membership(House,House_medium,defuzzified_house)
House_fit_high=fuzz.interp_membership(House,House_high,defuzzified_house)
House_fit_very_high=fuzz.interp_membership(House,House_very_high,defuzzified_house)

Applicant_fit_low=fuzz.interp_membership(Applicant,Applicant_low,defuzzified_applicant)
Applicant_fit_medium=fuzz.interp_membership(Applicant,Applicant_medium,defuzzified_applicant)
Applicant_fit_high=fuzz.interp_membership(Applicant,Applicant_high,defuzzified_applicant)

rule1_credit=prod(np.fmin(Income_fit_low,Interest_fit_medium),Credit_very_low)
rule2_credit=prod(np.fmin(Income_fit_low,Interest_fit_high),Credit_very_low)
rule3_credit=prod(np.fmin(Income_fit_medium,Interest_fit_high),Credit_low)
rule4_credit=prod(Applicant_fit_low,Credit_very_low)
rule5_credit=prod(House_fit_very_low,Credit_very_low)
rule6_credit=prod(np.fmin(Applicant_fit_medium,House_fit_very_low),Credit_low)
rule7_credit=prod(np.fmin(Applicant_fit_medium,House_fit_low),Credit_low)
rule8_credit=prod(np.fmin(Applicant_fit_medium,House_fit_medium),Credit_medium)
rule9_credit=prod(np.fmin(Applicant_fit_medium,House_fit_high),Credit_high)
rule10_credit=prod(np.fmin(Applicant_fit_medium,House_fit_very_high),Credit_high)
rule11_credit=prod(np.fmin(Applicant_fit_high,House_fit_very_low),Credit_low)
rule12_credit=prod(np.fmin(Applicant_fit_high,House_fit_low),Credit_medium)
rule13_credit=prod(np.fmin(Applicant_fit_high,House_fit_medium),Credit_high)
rule14_credit=prod(np.fmin(Applicant_fit_high,House_fit_high),Credit_high)
rule15_credit=prod(np.fmin(Applicant_fit_high,House_fit_very_high),Credit_very_high)

out_credit_very_low=np.fmax(np.fmax(rule1_credit,rule2_credit),np.fmax(rule4_credit,rule5_credit))
out_credit_low=np.fmax(np.fmax(rule3_credit,rule6_credit),np.fmax(rule7_credit,rule11_credit))
out_credit_medium=np.fmax(rule8_credit,rule12_credit)
out_credit_high=np.fmax(np.fmax(rule9_credit,rule10_credit),np.fmax(rule13_credit,rule14_credit))
out_credit_very_high=rule15_credit

output_credit=np.fmax(np.fmax(np.fmax(out_credit_low,out_credit_very_low),np.fmax(out_credit_medium,out_credit_high)),out_credit_very_high)
defuzzified_credit=fuzz.defuzz(Credit,output_credit,'centroid')
result_credit=fuzz.interp_membership(Credit,output_credit,defuzzified_credit)

Credit0=np.zeros_like(Credit)

fig, ax14 = plt.subplots(figsize=(7, 4))
ax14.plot(Credit,Credit_very_low,'r',linewidth=1,linestyle='--')
ax14.plot(Credit,Credit_low,'g',linewidth=1,linestyle='--')
ax14.plot(Credit,Credit_medium,'b',linewidth=1,linestyle='--')
ax14.plot(Credit,Credit_high,'y',linewidth=1,linestyle='--')
ax14.plot(Credit,Credit_very_high,'b',linewidth=1,linestyle='--')
ax14.fill_between(Credit,Credit0,output_credit,facecolor='Blue',alpha=0.7)
ax14.plot([defuzzified_credit,defuzzified_credit],[0,result_credit],'k',linewidth=2,alpha=0.9)
ax14.set_title('Girilen input değerlerine karşılık Credit için durulaştırılmış grafik')
plt.tight_layout()

print(f"credit  için bulanıklaştırılmış çıkış değeri :{round(defuzzified_credit,2)}")




