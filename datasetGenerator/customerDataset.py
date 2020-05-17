import csv
import random

with open('Datasets/customerPredictDataset.csv', mode='a') as customerData:
    for i in range(0,100):

        dataset_writer = csv.writer(customerData, delimiter=',')
        age  = round(random.uniform(0, 1.0),1)  # 25 - 60 yrs of age 60 corresponds to 60+ age
        #round(random.uniform(0, 1.1),1)
        totalAmt = round(random.uniform(0, 1.0),1) #Shopping cycle average amt spent
        profit = 0 # Company gets from customer opting for offer
        if(totalAmt < 0.5):
            profit = round(random.uniform(0, 0.50),1)
        else:
            profit =  round(random.uniform(0.5, 1.0),1)
        profitScore = round(random.uniform(0, 1.0),1) #Probability of customer taking the offer
        cashBack = round(0.1*age + 0.75*profit + 0.15*totalAmt,1) #Needs to be given back to customer opting offer
        financialStatus = totalAmt # Of the customer based on shopping cycle amount
        valueOfCustomer = round(0.1*age + profitScore*profit,1)
        dataset_writer.writerow([age,totalAmt,profit,profitScore,cashBack,financialStatus,valueOfCustomer])
