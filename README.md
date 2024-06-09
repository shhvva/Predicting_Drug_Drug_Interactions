# Predicting Drug Drug Interaction 💊

Mini Project done in third year by:
21R01A05N5 - Ashwitha K
21R01A05P0 - Shiva Shankar Mende
22R05A0522 - Sunny Joel
22R05A0526 - Kasi Vishwanath

Multi-drug therapies have widely been used to treat diseases, especially complex diseases such as cancer to improve the treatment effect and reduce the burden of patients. However, the adverse effects resulted from multi-drug therapies have also been observed, which may caused some serious complications and even the patient death. Therefore, identifying drug-drug interactions is helpful in contributing to improved treatment of diseases and reducing the difficulty of drug developments. Especially, it is very necessary to develop new computational methods for identifying DDIs.

## Prerequisites 📋

- Python 3.6.8
- MySQL
- XAMPP

## Installation 🛠️

1. Clone the repository or download the source code files.
2. Navigate to the Database directory.
3. Open the XAMPP server and start MySQL and Apache.
4. Click on admin on MySQL and upload the databse from Database directory.
5. Navigate to the DDI.
6. Open command prompt and install the requirements by `pip install -r requirements.txt` .
7. Now launch the website by `py manage.py runserver` ,the website is hosted on the localhost:8000 .

## Usage 📖

- Upon launching the application, 3 options login for remote users,admins and register an account.
- Modules :
1. Service Provider
In this module, the Service Provider has to login by using valid user name and password. After login successful he can do some operations such as          
Login, Train and Test Drugs Data Sets, View Drugs Trained and Tested Accuracy in Bar Chart, View Drugs Trained and Tested Accuracy Results, 
View Drug to Drug Interaction Predicted Details, Find Drug to Drug Interaction Predicted Ratio, Download Drug to Drug Interaction, View Drug to Drug Interaction Predicted Ratio Results, View All Remote Users.

2. View and Authorize Users
In this module, the admin can view the list of users who all registered. In this, the admin can view the user’s details such as, user name, email, address and admin authorizes the users.

3. Remote User
In this module, there are n numbers of users are present. User should register before doing any operations. Once user registers, their details will be stored to the database.  After registration successful, he has to login by using authorized user name and password. Once Login is successful user will do some operations like  REGISTER AND LOGIN, PREDICT DRUG TO DRUG INTERACTION TYPE, VIEW YOUR PROFILE.

## Algorithms

- Decision tree classifiers
- Gradient boosting 
- K-Nearest Neighbors (KNN)
- Logistic regression Classifiers
- Naïve Bayes
- Random Forest 
- SVM 

## Conclusion

Multi-drug therapies have widely been used to treat diseases, especially complex diseases such as cancer to improve the treatment effect and reduce the burden of patients. However, the adverse effects resulted from multi-drug therapies have also been observed, which may caused some serious complications and even the patient death. Therefore, identifying drug-drug interactions is helpful in contributing to improved treatment of diseases and reducing the difficulty of drug developments. Especially, it is very necessary to develop new computational methods for identifying DDIs.

## Contact us
21R01A05N5 - Ashwitha K - 21R01A05N5@cmritonline.ac.in
21R01A05P0 - Shiva Shankar Mende - 21R01A05P0@cmritonline.ac.in
22R05A0522 - Sunny Joel - 22R05A0522@cmritonline.ac.in
22R05A0526 - Kasi Vishwanath - 22R05A0526@cmritonline.ac.in