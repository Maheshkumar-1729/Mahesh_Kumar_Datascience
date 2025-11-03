# 1)A company offers dearness allowance (DA) of 40% of basic pay and house rent allowance (HRA) of 10% of basic pay. Input basic pay of an employee, calculate his/her DA, HRA and Gross pay (Gross = Basic Pay + DA+ HRA).

# Input: Basic Pay
basic_pay = float(input("Enter Basic Pay: "))

# Fixed percentages
DA_percent = 40
HRA_percent = 10

# Calculations
DA = (DA_percent / 100) * basic_pay
HRA = (HRA_percent / 100) * basic_pay
gross_pay = basic_pay + DA + HRA

# Output
print(f"\nDearness Allowance (DA): ₹{DA:.2f}")
print(f"House Rent Allowance (HRA): ₹{HRA:.2f}")
print(f"Gross Pay: ₹{gross_pay:.2f}")


# 1(a): DA and HRA percentages as inputs

# Input: Basic Pay and percentages
basic_pay = float(input("Enter Basic Pay: "))
DA_percent = float(input("Enter DA percentage: "))
HRA_percent = float(input("Enter HRA percentage: "))

# Calculations
DA = (DA_percent / 100) * basic_pay
HRA = (HRA_percent / 100) * basic_pay
gross_pay = basic_pay + DA + HRA

# Output
print(f"\nDearness Allowance (DA): ₹{DA:.2f}")
print(f"House Rent Allowance (HRA): ₹{HRA:.2f}")
print(f"Gross Pay: ₹{gross_pay:.2f}")

# 1(b): Using a user-defined function

# Function to calculate Gross Pay
def calculate_gross_pay(basic_pay, da_percent, hra_percent):
    da = (da_percent / 100) * basic_pay
    hra = (hra_percent / 100) * basic_pay
    gross = basic_pay + da + hra
    return gross

# Input
basic_pay = float(input("Enter Basic Pay: "))
da_percent = float(input("Enter DA percentage: "))
hra_percent = float(input("Enter HRA percentage: "))

# Function call
gross_pay = calculate_gross_pay(basic_pay, da_percent, hra_percent)

# Output
print(f"\nGross Pay: ₹{gross_pay:.2f}")







